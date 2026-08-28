from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


# ---------- Tensor / PIL helpers ----------

def _tensor_image_to_pil(image: torch.Tensor, index: int = 0) -> Image.Image:
    if image.ndim == 3:
        frame = image
    elif image.ndim == 4:
        frame = image[index % image.shape[0]]
    else:
        raise ValueError(f"Expected IMAGE tensor [H,W,C] or [B,H,W,C], got {tuple(image.shape)}")

    arr = frame.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.rint(arr * 255.0).astype(np.uint8)

    if arr.shape[-1] >= 4:
        return Image.fromarray(arr[..., :4], "RGBA")
    return Image.fromarray(arr[..., :3], "RGB")


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    # Outputs are expected to share dimensions because the mockup canvas is fixed.
    return torch.cat(images, dim=0)


# ---------- Sticker extraction ----------

def _edge_connected_white_foreground(
    image: Image.Image,
    threshold: int,
    feather: float,
) -> Image.Image:
    """
    Return foreground mask (L, 0..255).

    Pixels that are white-ish AND connected to an outer edge are considered
    background. Disconnected white regions inside the art are preserved.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    candidate = np.all(rgb >= int(threshold), axis=2).astype(np.uint8) * 255
    work = Image.fromarray(candidate, "L")
    draw = ImageDraw.Draw(work)
    w, h = work.size

    # Seed flood-fill from many edge points. Usually the first seed fills the
    # whole outer background; extra seeds handle split white regions along edges.
    stride = max(1, min(w, h) // 128)
    edge_points = []
    for x in range(0, w, stride):
        edge_points.append((x, 0))
        edge_points.append((x, h - 1))
    for y in range(0, h, stride):
        edge_points.append((0, y))
        edge_points.append((w - 1, y))

    # Ensure exact corners/endpoints are covered.
    edge_points.extend([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    for xy in edge_points:
        if work.getpixel(xy) == 255:
            ImageDraw.floodfill(work, xy, 128, thresh=0)

    filled = np.asarray(work, dtype=np.uint8)
    # 128 = edge-connected candidate background. Everything else is foreground.
    fg = np.where(filled == 128, 0, 255).astype(np.uint8)
    mask = Image.fromarray(fg, "L")

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return mask


def _all_white_foreground(image: Image.Image, threshold: int, feather: float) -> Image.Image:
    """Simpler mode: every white-ish pixel becomes transparent."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    t = float(threshold)
    # Soft transition from threshold-12 to threshold for anti-aliased edges.
    low = max(0.0, t - 12.0)
    whiteness = np.min(rgb, axis=2)
    bg = np.clip((whiteness - low) / max(1.0, t - low), 0.0, 1.0)
    fg = np.rint((1.0 - bg) * 255.0).astype(np.uint8)
    mask = Image.fromarray(fg, "L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return mask


def _foreground_mask(
    image: Image.Image,
    mode: str,
    threshold: int,
    feather: float,
) -> Image.Image:
    if mode == "all white":
        return _all_white_foreground(image, threshold, feather)
    return _edge_connected_white_foreground(image, threshold, feather)


def _expand_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.copy()

    # Pillow requires an odd filter size. Very wide borders are done in chunks
    # to keep MaxFilter memory/runtime reasonable.
    out = mask
    remaining = int(radius)
    while remaining > 0:
        step = min(remaining, 32)
        out = out.filter(ImageFilter.MaxFilter(step * 2 + 1))
        remaining -= step
    return out


def _crop_with_padding(
    art: Image.Image,
    subject_mask: Image.Image,
    sticker_mask: Image.Image,
    padding: int,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    bbox = sticker_mask.getbbox()
    if bbox is None:
        raise ValueError("No sticker foreground detected. Lower white_threshold or use a different removal mode.")

    l, t, r, b = bbox
    l = max(0, l - padding)
    t = max(0, t - padding)
    r = min(art.width, r + padding)
    b = min(art.height, b + padding)
    box = (l, t, r, b)
    return art.crop(box), subject_mask.crop(box), sticker_mask.crop(box)


def _make_sticker(
    image: Image.Image,
    removal_mode: str,
    white_threshold: int,
    edge_feather: float,
    border_size: int,
    border_softness: float,
    border_rgb: Tuple[int, int, int],
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    art = image.convert("RGB")
    subject_mask = _foreground_mask(art, removal_mode, white_threshold, edge_feather)
    sticker_mask = _expand_mask(subject_mask, border_size)

    if border_softness > 0:
        sticker_mask = sticker_mask.filter(ImageFilter.GaussianBlur(radius=float(border_softness)))

    art, subject_mask, sticker_mask = _crop_with_padding(
        art,
        subject_mask,
        sticker_mask,
        padding=max(4, border_size // 2 + 2),
    )

    # White/coloured border underneath, original art clipped above it.
    sticker_rgba = Image.new("RGBA", art.size, (*border_rgb, 0))
    border_layer = Image.new("RGBA", art.size, (*border_rgb, 255))
    sticker_rgba.paste(border_layer, (0, 0), sticker_mask)

    art_rgba = art.convert("RGBA")
    art_rgba.putalpha(subject_mask)
    sticker_rgba = Image.alpha_composite(sticker_rgba, art_rgba)

    return sticker_rgba, subject_mask, sticker_mask


# ---------- Mockup composition ----------

def _contain_size(src_w: int, src_h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    scale = min(max_w / max(1, src_w), max_h / max(1, src_h))
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


def _transform_sticker(
    sticker: Image.Image,
    sticker_mask: Image.Image,
    target_w: int,
    target_h: int,
    rotation: float,
) -> Tuple[Image.Image, Image.Image]:
    new_w, new_h = _contain_size(sticker.width, sticker.height, target_w, target_h)
    sticker = sticker.resize((new_w, new_h), Image.Resampling.LANCZOS)
    sticker_mask = sticker_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if abs(rotation) > 0.001:
        sticker = sticker.rotate(
            float(rotation),
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        sticker_mask = sticker_mask.rotate(
            float(rotation),
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
    return sticker, sticker_mask


def _make_canvas(
    background: Optional[torch.Tensor],
    background_index: int,
    canvas_mode: str,
    width: int,
    height: int,
) -> Image.Image:
    if canvas_mode == "background" and background is not None:
        bg = _tensor_image_to_pil(background, background_index).convert("RGB")
        # Cover crop background to canvas dimensions.
        src_ratio = bg.width / max(1, bg.height)
        dst_ratio = width / max(1, height)
        if src_ratio > dst_ratio:
            new_h = height
            new_w = round(height * src_ratio)
        else:
            new_w = width
            new_h = round(width / max(src_ratio, 1e-6))
        bg = bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = max(0, (new_w - width) // 2)
        top = max(0, (new_h - height) // 2)
        return bg.crop((left, top, left + width, top + height))

    if canvas_mode == "warm off-white":
        return Image.new("RGB", (width, height), (247, 245, 239))
    if canvas_mode == "light grey":
        return Image.new("RGB", (width, height), (238, 238, 238))
    return Image.new("RGB", (width, height), (255, 255, 255))


def _shadow_from_mask(mask: Image.Image, blur: float, opacity: float) -> Image.Image:
    shadow_alpha = mask
    if blur > 0:
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    opacity = max(0.0, min(1.0, float(opacity)))
    shadow_alpha = shadow_alpha.point(lambda p: int(round(p * opacity)))
    shadow = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    return shadow


def _paste_clipped(base: Image.Image, overlay: Image.Image, xy: Tuple[int, int]) -> None:
    """Paste RGBA overlay even if partly outside the canvas."""
    x, y = xy
    bw, bh = base.size
    ow, oh = overlay.size

    left = max(0, x)
    top = max(0, y)
    right = min(bw, x + ow)
    bottom = min(bh, y + oh)
    if right <= left or bottom <= top:
        return

    crop = overlay.crop((left - x, top - y, right - x, bottom - y))
    base.alpha_composite(crop, (left, top))


class OPSFeaturedStickerMockup:
    """Create a sticker-style product/featured image from artwork on white."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artwork": ("IMAGE",),
                "remove_background": (["edge white", "all white"],),
                "white_threshold": ("INT", {"default": 245, "min": 180, "max": 255, "step": 1}),
                "edge_feather": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 12.0, "step": 0.1}),
                "border_size": ("INT", {"default": 22, "min": 0, "max": 160, "step": 1}),
                "border_softness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 12.0, "step": 0.1}),
                "canvas_mode": (["background", "white", "warm off-white", "light grey"],),
                "canvas_width": ("INT", {"default": 1600, "min": 256, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 1600, "min": 256, "max": 8192, "step": 8}),
                "sticker_size_percent": ("FLOAT", {"default": 72.0, "min": 5.0, "max": 150.0, "step": 0.5}),
                "x_percent": ("FLOAT", {"default": 50.0, "min": -25.0, "max": 125.0, "step": 0.5}),
                "y_percent": ("FLOAT", {"default": 50.0, "min": -25.0, "max": 125.0, "step": 0.5}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "shadow_opacity": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_blur": ("FLOAT", {"default": 22.0, "min": 0.0, "max": 120.0, "step": 1.0}),
                "shadow_x": ("INT", {"default": 14, "min": -256, "max": 256, "step": 1}),
                "shadow_y": ("INT", {"default": 18, "min": -256, "max": 256, "step": 1}),
            },
            "optional": {
                "background": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mockup", "sticker_rgb", "sticker_mask")
    FUNCTION = "make_mockup"
    CATEGORY = "OPS/Mockups"
    DESCRIPTION = (
        "Removes an outer white background from artwork, adds a sticker border and shadow, "
        "then places it onto a featured-image/mockup canvas."
    )

    def make_mockup(
        self,
        artwork: torch.Tensor,
        remove_background: str,
        white_threshold: int,
        edge_feather: float,
        border_size: int,
        border_softness: float,
        canvas_mode: str,
        canvas_width: int,
        canvas_height: int,
        sticker_size_percent: float,
        x_percent: float,
        y_percent: float,
        rotation: float,
        shadow_opacity: float,
        shadow_blur: float,
        shadow_x: int,
        shadow_y: int,
        background: Optional[torch.Tensor] = None,
    ):
        batch = artwork.shape[0] if artwork.ndim == 4 else 1
        mockups = []
        sticker_rgbs = []
        sticker_masks = []

        # Sticker output uses a predictable square canvas so batch outputs remain stackable.
        sticker_out_size = max(512, min(2048, max(int(canvas_width), int(canvas_height))))

        for i in range(batch):
            art = _tensor_image_to_pil(artwork, i).convert("RGB")
            sticker, _, final_mask = _make_sticker(
                art,
                removal_mode=remove_background,
                white_threshold=int(white_threshold),
                edge_feather=float(edge_feather),
                border_size=int(border_size),
                border_softness=float(border_softness),
                border_rgb=(255, 255, 255),
            )

            # Scale sticker relative to canvas, preserving aspect ratio.
            target_w = max(1, round(canvas_width * sticker_size_percent / 100.0))
            target_h = max(1, round(canvas_height * sticker_size_percent / 100.0))
            placed_sticker, placed_mask = _transform_sticker(
                sticker,
                final_mask,
                target_w,
                target_h,
                rotation,
            )

            canvas = _make_canvas(background, i, canvas_mode, canvas_width, canvas_height).convert("RGBA")
            cx = round(canvas_width * x_percent / 100.0)
            cy = round(canvas_height * y_percent / 100.0)
            x = cx - placed_sticker.width // 2
            y = cy - placed_sticker.height // 2

            if shadow_opacity > 0.0:
                shadow = _shadow_from_mask(placed_mask, shadow_blur, shadow_opacity)
                _paste_clipped(canvas, shadow, (x + int(shadow_x), y + int(shadow_y)))

            _paste_clipped(canvas, placed_sticker, (x, y))
            mockups.append(_pil_rgb_to_tensor(canvas.convert("RGB")))

            # Separate sticker output on neutral black RGB + MASK. Consumers should use
            # sticker_mask for alpha/compositing; black is outside the mask only.
            sw, sh = _contain_size(
                sticker.width,
                sticker.height,
                round(sticker_out_size * 0.90),
                round(sticker_out_size * 0.90),
            )
            sticker_preview = sticker.resize((sw, sh), Image.Resampling.LANCZOS)
            sticker_mask_preview = final_mask.resize((sw, sh), Image.Resampling.LANCZOS)
            px = (sticker_out_size - sw) // 2
            py = (sticker_out_size - sh) // 2

            rgb_canvas = Image.new("RGB", (sticker_out_size, sticker_out_size), (0, 0, 0))
            rgb_canvas.paste(sticker_preview.convert("RGB"), (px, py), sticker_preview.getchannel("A"))
            mask_canvas = Image.new("L", (sticker_out_size, sticker_out_size), 0)
            mask_canvas.paste(sticker_mask_preview, (px, py))

            sticker_rgbs.append(_pil_rgb_to_tensor(rgb_canvas))
            sticker_masks.append(_pil_mask_to_tensor(mask_canvas))

        return (
            _stack_images(mockups),
            _stack_images(sticker_rgbs),
            torch.cat(sticker_masks, dim=0),
        )


NODE_CLASS_MAPPINGS = {
    "OPSFeaturedStickerMockup": OPSFeaturedStickerMockup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPSFeaturedStickerMockup": "OPS Featured Sticker Mockup",
}
