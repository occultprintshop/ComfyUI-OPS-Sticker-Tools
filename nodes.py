from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


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
    return torch.cat(images, dim=0)


# ---------- Generic mask helpers ----------

def _hard_mask(mask: Image.Image, threshold: int = 96) -> Image.Image:
    m = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=0.45))
    return m.point(lambda p: 255 if p >= threshold else 0)


def _soft_grow_or_shrink(mask: Image.Image, pixels: int) -> Image.Image:
    """
    Rounded mask inset/outset without square MaxFilter geometry.

    Positive pixels shrink/inset the active area.
    Negative pixels expand/outset the active area.
    """
    px = int(pixels)
    if px == 0:
        return mask.convert("L").copy()

    hard = _hard_mask(mask, 128)
    radius = max(0.8, abs(px) * 0.72)
    blurred = hard.filter(ImageFilter.GaussianBlur(radius=radius))

    if px > 0:
        out = blurred.point(lambda p: 255 if p >= 205 else 0)
    else:
        out = blurred.point(lambda p: 255 if p >= 50 else 0)

    return out.filter(ImageFilter.GaussianBlur(radius=0.45))


def _approx_remove_small_islands(mask: Image.Image, minimum_size: int) -> Image.Image:
    """
    Approximate small-island suppression for the CUT silhouette.

    minimum_size is an approximate diameter in pixels, not component area.
    It intentionally acts only on the cut source by default so delicate
    watercolour splashes can remain visible in the artwork.
    """
    size = max(0, int(minimum_size))
    if size <= 1:
        return _hard_mask(mask)

    kernel = min(31, max(3, size if size % 2 == 1 else size + 1))
    hard = _hard_mask(mask)
    opened = hard.filter(ImageFilter.MinFilter(kernel)).filter(ImageFilter.MaxFilter(kernel))
    return opened


def _mask_from_image(image: Image.Image, channel: str, invert: bool) -> Image.Image:
    rgb = image.convert("RGB")
    if channel == "red":
        mask = rgb.getchannel("R")
    elif channel == "green":
        mask = rgb.getchannel("G")
    elif channel == "blue":
        mask = rgb.getchannel("B")
    else:
        arr = np.asarray(rgb, dtype=np.float32)
        lum = np.rint(arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114).astype(np.uint8)
        mask = Image.fromarray(lum, "L")

    if invert:
        mask = ImageOps.invert(mask)
    return mask


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
    w, h = work.size

    stride = max(1, min(w, h) // 128)
    edge_points = []
    for x in range(0, w, stride):
        edge_points.append((x, 0))
        edge_points.append((x, h - 1))
    for y in range(0, h, stride):
        edge_points.append((0, y))
        edge_points.append((w - 1, y))
    edge_points.extend([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    for xy in edge_points:
        if work.getpixel(xy) == 255:
            ImageDraw.floodfill(work, xy, 128, thresh=0)

    filled = np.asarray(work, dtype=np.uint8)
    fg = np.where(filled == 128, 0, 255).astype(np.uint8)
    mask = Image.fromarray(fg, "L")

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return mask


def _all_white_foreground(image: Image.Image, threshold: int, feather: float) -> Image.Image:
    """Simpler mode: every white-ish pixel becomes transparent."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    t = float(threshold)
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


def _merge_close_fragments(
    mask: Image.Image,
    border_radius: int,
    merge_strength: float,
) -> Image.Image:
    """Join nearby paint flecks/splashes into a more practical cut shape."""
    hard = _hard_mask(mask)
    strength = max(0.0, float(merge_strength))
    if strength <= 0.001:
        return hard

    merge_radius = max(
        1.0,
        min(24.0, float(border_radius) * 0.28 * strength),
    )
    if merge_radius <= 1.05:
        return hard

    bridge = hard.filter(ImageFilter.GaussianBlur(radius=merge_radius))
    bridge = bridge.point(lambda p: 255 if p >= 44 else 0)

    bridge = bridge.filter(ImageFilter.GaussianBlur(radius=merge_radius * 0.82))
    bridge = bridge.point(lambda p: 255 if p >= 176 else 0)

    return ImageChops.lighter(hard, bridge)


def _expand_mask(
    mask: Image.Image,
    radius: int,
    merge_strength: float = 1.0,
) -> Image.Image:
    """Create a rounded, watercolour-friendly sticker silhouette."""
    if radius <= 0:
        return mask.copy()

    clean = _merge_close_fragments(mask, radius, merge_strength)

    sigma = max(0.8, float(radius) * 0.72)
    expanded = clean.filter(ImageFilter.GaussianBlur(radius=sigma))
    expanded = expanded.point(lambda p: 255 if p >= 55 else 0)

    return expanded.filter(ImageFilter.GaussianBlur(radius=0.75))


def _crop_with_padding(
    art: Image.Image,
    subject_mask: Image.Image,
    sticker_mask: Image.Image,
    padding: int,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    # Include any preserved loose artwork in the crop even when it has been
    # intentionally excluded from the physical cut silhouette.
    crop_extent = ImageChops.lighter(
        sticker_mask.convert("L"),
        subject_mask.convert("L"),
    )
    bbox = crop_extent.getbbox()
    if bbox is None:
        raise ValueError(
            "No sticker foreground detected. Raise white_threshold to preserve paler artwork "
            "or use a different removal mode."
        )

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
    merge_strength: float = 1.0,
    minimum_island_size: int = 0,
    preserve_loose_splashes: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    art = image.convert("RGB")
    original_subject_mask = _foreground_mask(
        art,
        removal_mode,
        white_threshold,
        edge_feather,
    )

    # Cutline controls are deliberately separate from artwork visibility.
    # This lets the user suppress tiny cut islands while still keeping loose
    # watercolour marks visible when preserve_loose_splashes is enabled.
    cut_source_mask = _approx_remove_small_islands(
        original_subject_mask,
        minimum_island_size,
    )

    if preserve_loose_splashes:
        visible_subject_mask = original_subject_mask
    else:
        visible_subject_mask = ImageChops.multiply(
            original_subject_mask.convert("L"),
            cut_source_mask.convert("L"),
        )

    work_padding = max(
        16,
        int(border_size) * 2
        + int(round(float(border_softness) * 4.0))
        + int(round(float(border_size) * max(0.0, float(merge_strength)) * 0.4))
        + 8,
    )

    art = ImageOps.expand(art, border=work_padding, fill=(255, 255, 255))
    visible_subject_mask = ImageOps.expand(visible_subject_mask, border=work_padding, fill=0)
    cut_source_mask = ImageOps.expand(cut_source_mask, border=work_padding, fill=0)

    sticker_mask = _expand_mask(
        cut_source_mask,
        border_size,
        merge_strength=merge_strength,
    )

    if border_softness > 0:
        sticker_mask = sticker_mask.filter(
            ImageFilter.GaussianBlur(radius=float(border_softness))
        )

    art, visible_subject_mask, sticker_mask = _crop_with_padding(
        art,
        visible_subject_mask,
        sticker_mask,
        padding=max(4, border_size // 2 + 2),
    )

    sticker_rgba = Image.new("RGBA", art.size, (*border_rgb, 0))
    border_layer = Image.new("RGBA", art.size, (*border_rgb, 255))
    sticker_rgba.paste(border_layer, (0, 0), sticker_mask)

    art_rgba = art.convert("RGBA")
    art_rgba.putalpha(visible_subject_mask)
    sticker_rgba = Image.alpha_composite(sticker_rgba, art_rgba)

    return sticker_rgba, visible_subject_mask, sticker_mask


# ---------- Shared composition helpers ----------

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


# ---------- Product template helpers ----------

def _fit_artwork_to_box(
    artwork: Image.Image,
    box_w: int,
    box_h: int,
    fit_mode: str,
    scale_percent: float,
    x_offset: int,
    y_offset: int,
    rotation: float,
) -> Image.Image:
    """Return an RGB artwork tile exactly box_w x box_h."""
    src = artwork.convert("RGB")
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))
    scale_factor = max(0.01, float(scale_percent) / 100.0)

    if fit_mode == "stretch":
        target_w = max(1, round(box_w * scale_factor))
        target_h = max(1, round(box_h * scale_factor))
    else:
        if fit_mode == "contain":
            base_scale = min(box_w / max(1, src.width), box_h / max(1, src.height))
        else:
            base_scale = max(box_w / max(1, src.width), box_h / max(1, src.height))
        base_scale *= scale_factor
        target_w = max(1, round(src.width * base_scale))
        target_h = max(1, round(src.height * base_scale))

    resized = src.resize((target_w, target_h), Image.Resampling.LANCZOS)

    if abs(float(rotation)) > 0.001:
        resized = resized.rotate(
            float(rotation),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )

    tile = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    x = (box_w - resized.width) // 2 + int(x_offset)
    y = (box_h - resized.height) // 2 + int(y_offset)

    left = max(0, x)
    top = max(0, y)
    right = min(box_w, x + resized.width)
    bottom = min(box_h, y + resized.height)
    if right > left and bottom > top:
        crop = resized.crop((left - x, top - y, right - x, bottom - y))
        tile.paste(crop, (left, top))

    return tile


def _apply_template_shading(
    fitted: Image.Image,
    template: Image.Image,
    strength: float,
) -> Image.Image:
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return fitted

    art = np.asarray(fitted.convert("RGB"), dtype=np.float32)
    tmpl = np.asarray(template.convert("RGB"), dtype=np.float32)
    lum = tmpl[..., 0] * 0.299 + tmpl[..., 1] * 0.587 + tmpl[..., 2] * 0.114

    shade = (1.0 - s) + s * (lum / 255.0)
    out = np.clip(art * shade[..., None], 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def _template_mockup_single(
    artwork: Image.Image,
    template: Image.Image,
    mask_image: Image.Image,
    fit_mode: str,
    mask_channel: str,
    invert_mask: bool,
    scale_percent: float,
    x_offset: int,
    y_offset: int,
    rotation: float,
    mask_inset: int,
    edge_blend: float,
    shading_strength: float,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    template_rgb = template.convert("RGB")
    mask = _mask_from_image(mask_image, mask_channel, invert_mask)

    if mask_inset != 0:
        mask = _soft_grow_or_shrink(mask, int(mask_inset))

    geometry_mask = _hard_mask(mask, 32)
    bbox = geometry_mask.getbbox()
    if bbox is None:
        raise ValueError("Template mask contains no active placement area.")

    l, t, r, b = bbox
    box_w = max(1, r - l)
    box_h = max(1, b - t)

    tile = _fit_artwork_to_box(
        artwork,
        box_w,
        box_h,
        fit_mode,
        scale_percent,
        x_offset,
        y_offset,
        rotation,
    )

    fitted_full = Image.new("RGB", template_rgb.size, (255, 255, 255))
    fitted_full.paste(tile, (l, t))

    shaded = _apply_template_shading(
        fitted_full,
        template_rgb,
        shading_strength,
    )

    effective_mask = mask.convert("L")
    if edge_blend > 0:
        effective_mask = effective_mask.filter(
            ImageFilter.GaussianBlur(radius=float(edge_blend))
        )

    result = Image.composite(shaded, template_rgb, effective_mask)
    return result, fitted_full, effective_mask


# ---------- Nodes ----------

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
                "merge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "minimum_island_size": ("INT", {"default": 0, "min": 0, "max": 31, "step": 2}),
                "preserve_loose_splashes": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mockup", "sticker_rgb", "sticker_mask")
    FUNCTION = "make_mockup"
    CATEGORY = "OPS/Mockups"
    DESCRIPTION = (
        "Removes an outer white background from artwork, adds a sticker border and shadow, "
        "and provides optional watercolour cut-shape controls."
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
        merge_strength: float = 1.0,
        minimum_island_size: int = 0,
        preserve_loose_splashes: bool = True,
    ):
        batch = artwork.shape[0] if artwork.ndim == 4 else 1
        mockups = []
        sticker_rgbs = []
        sticker_masks = []

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
                merge_strength=float(merge_strength),
                minimum_island_size=int(minimum_island_size),
                preserve_loose_splashes=bool(preserve_loose_splashes),
            )

            target_w = max(1, round(canvas_width * sticker_size_percent / 100.0))
            target_h = max(1, round(canvas_height * sticker_size_percent / 100.0))
            placed_sticker, placed_mask = _transform_sticker(
                sticker,
                final_mask,
                target_w,
                target_h,
                rotation,
            )

            canvas = _make_canvas(
                background,
                i,
                canvas_mode,
                canvas_width,
                canvas_height,
            ).convert("RGBA")
            cx = round(canvas_width * x_percent / 100.0)
            cy = round(canvas_height * y_percent / 100.0)
            x = cx - placed_sticker.width // 2
            y = cy - placed_sticker.height // 2

            if shadow_opacity > 0.0:
                shadow = _shadow_from_mask(
                    placed_mask,
                    shadow_blur,
                    shadow_opacity,
                )
                _paste_clipped(
                    canvas,
                    shadow,
                    (x + int(shadow_x), y + int(shadow_y)),
                )

            _paste_clipped(canvas, placed_sticker, (x, y))
            mockups.append(_pil_rgb_to_tensor(canvas.convert("RGB")))

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
            rgb_canvas.paste(
                sticker_preview.convert("RGB"),
                (px, py),
                sticker_preview.getchannel("A"),
            )
            mask_canvas = Image.new("L", (sticker_out_size, sticker_out_size), 0)
            mask_canvas.paste(sticker_mask_preview, (px, py))

            sticker_rgbs.append(_pil_rgb_to_tensor(rgb_canvas))
            sticker_masks.append(_pil_mask_to_tensor(mask_canvas))

        return (
            _stack_images(mockups),
            _stack_images(sticker_rgbs),
            torch.cat(sticker_masks, dim=0),
        )


class OPSProductTemplateMockup:
    """
    Put complete artwork into a supplied product template/mask.

    Designed for greeting cards, badges, apparel print areas and other
    deterministic OPS mockups where the artwork itself should remain exact.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artwork": ("IMAGE",),
                "template": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "fit_mode": (["cover", "contain", "stretch"],),
                "mask_channel": (["luminance", "red", "green", "blue"],),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "scale_percent": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 250.0, "step": 0.5}),
                "x_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.25}),
                "mask_inset": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "edge_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 32.0, "step": 0.25}),
                "shading_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("mockup", "fitted_artwork", "effective_mask")
    FUNCTION = "make_template_mockup"
    CATEGORY = "OPS/Mockups"
    DESCRIPTION = (
        "Fits complete artwork into a supplied template mask without removing white. "
        "Useful for cards, badges, shirts, hoodies, totes and other product mockups."
    )

    def make_template_mockup(
        self,
        artwork: torch.Tensor,
        template: torch.Tensor,
        mask_image: torch.Tensor,
        fit_mode: str,
        mask_channel: str,
        invert_mask: bool,
        scale_percent: float,
        x_offset: int,
        y_offset: int,
        rotation: float,
        mask_inset: int,
        edge_blend: float,
        shading_strength: float,
    ):
        art_batch = artwork.shape[0] if artwork.ndim == 4 else 1
        template_batch = template.shape[0] if template.ndim == 4 else 1
        mask_batch = mask_image.shape[0] if mask_image.ndim == 4 else 1
        batch = max(art_batch, template_batch, mask_batch)

        mockups = []
        fitted = []
        masks = []

        expected_size = None
        for i in range(batch):
            art_pil = _tensor_image_to_pil(artwork, i)
            template_pil = _tensor_image_to_pil(template, i)
            mask_pil = _tensor_image_to_pil(mask_image, i)

            result, fitted_full, effective_mask = _template_mockup_single(
                artwork=art_pil,
                template=template_pil,
                mask_image=mask_pil,
                fit_mode=fit_mode,
                mask_channel=mask_channel,
                invert_mask=bool(invert_mask),
                scale_percent=float(scale_percent),
                x_offset=int(x_offset),
                y_offset=int(y_offset),
                rotation=float(rotation),
                mask_inset=int(mask_inset),
                edge_blend=float(edge_blend),
                shading_strength=float(shading_strength),
            )

            if expected_size is None:
                expected_size = result.size
            elif result.size != expected_size:
                raise ValueError(
                    "Template batch images must share the same dimensions for ComfyUI batching."
                )

            mockups.append(_pil_rgb_to_tensor(result))
            fitted.append(_pil_rgb_to_tensor(fitted_full))
            masks.append(_pil_mask_to_tensor(effective_mask))

        return (
            _stack_images(mockups),
            _stack_images(fitted),
            torch.cat(masks, dim=0),
        )


NODE_CLASS_MAPPINGS = {
    "OPSFeaturedStickerMockup": OPSFeaturedStickerMockup,
    "OPSProductTemplateMockup": OPSProductTemplateMockup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPSFeaturedStickerMockup": "OPS Featured Sticker Mockup",
    "OPSProductTemplateMockup": "OPS Product Template Mockup",
}
