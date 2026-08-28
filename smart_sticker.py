from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


def _bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Euclidean distance between two non-overlapping/overlapping bounding boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def _connected_components(binary_mask: Image.Image):
    """Return 8-connected components without requiring OpenCV or SciPy."""
    arr = np.asarray(binary_mask.convert("L"), dtype=np.uint8) > 0
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    components = []

    ys, xs = np.nonzero(arr)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue

        q = deque([(sy, sx)])
        visited[sy, sx] = True
        pixels = []
        min_x = max_x = sx
        min_y = max_y = sy

        while q:
            y, x = q.popleft()
            pixels.append(y * w + x)
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y

            y0 = max(0, y - 1)
            y1 = min(h - 1, y + 1)
            x0 = max(0, x - 1)
            x1 = min(w - 1, x + 1)
            for ny in range(y0, y1 + 1):
                for nx in range(x0, x1 + 1):
                    if ny == y and nx == x:
                        continue
                    if arr[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

        components.append(
            {
                "area": len(pixels),
                "pixels": pixels,
                "bbox": (min_x, min_y, max_x + 1, max_y + 1),
            }
        )

    return components, (w, h)


def _smart_component_masks(
    nodes_module,
    mask: Image.Image,
    minimum_size: int,
    border_radius: int,
) -> Tuple[Image.Image, Image.Image]:
    """
    Build separate masks for the physical cutline and visible loose artwork.

    Large components always survive. Tiny components survive the cutline only
    when they are close to substantial artwork; a slightly wider allowance is
    used for visible watercolour splashes. This removes distant speck/noise
    without destroying meaningful nearby stars, droplets and paint marks.
    """
    hard = nodes_module._hard_mask(mask)
    size = max(0, int(minimum_size))
    if size <= 1:
        return hard, hard

    components, (w, h) = _connected_components(hard)
    if not components:
        empty = Image.new("L", (w, h), 0)
        return empty, empty

    # Treat the UI value as an approximate component diameter. Squaring it
    # gives intuitive behaviour: 6 ~= 36 px, 10 ~= 100 px.
    area_threshold = max(4, size * size)
    largest = max(range(len(components)), key=lambda i: components[i]["area"])

    substantial = {
        i
        for i, comp in enumerate(components)
        if comp["area"] >= area_threshold
    }
    substantial.add(largest)
    anchor_boxes = [components[i]["bbox"] for i in substantial]

    # The cutline allowance is tighter than the visible-art allowance. A loose
    # splash may remain visible without forcing its own white halo.
    cut_proximity = max(
        8,
        min(96, int(round(max(float(border_radius) * 1.35, float(size) * 2.5)))),
    )
    visible_proximity = max(
        cut_proximity,
        min(128, int(round(cut_proximity * 1.25))),
    )

    keep_cut = set(substantial)
    keep_visible = set(substantial)

    for i, comp in enumerate(components):
        if i in substantial:
            continue
        distance = min(_bbox_distance(comp["bbox"], box) for box in anchor_boxes)
        if distance <= visible_proximity:
            keep_visible.add(i)
        if distance <= cut_proximity:
            keep_cut.add(i)

    cut_arr = np.zeros(h * w, dtype=np.uint8)
    visible_arr = np.zeros(h * w, dtype=np.uint8)

    for i in keep_cut:
        cut_arr[components[i]["pixels"]] = 255
    for i in keep_visible:
        visible_arr[components[i]["pixels"]] = 255

    cut_mask = Image.fromarray(cut_arr.reshape((h, w)), "L")
    visible_mask = Image.fromarray(visible_arr.reshape((h, w)), "L")
    return cut_mask, visible_mask


def _make_sticker_smart(
    nodes_module,
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
):
    art = image.convert("RGB")
    original_subject_mask = nodes_module._foreground_mask(
        art,
        removal_mode,
        white_threshold,
        edge_feather,
    )

    cut_component_mask, visible_component_mask = _smart_component_masks(
        nodes_module,
        original_subject_mask,
        minimum_island_size,
        border_size,
    )

    cut_source_mask = ImageChops.multiply(
        original_subject_mask.convert("L"),
        cut_component_mask.convert("L"),
    )

    if preserve_loose_splashes:
        visible_subject_mask = ImageChops.multiply(
            original_subject_mask.convert("L"),
            visible_component_mask.convert("L"),
        )
    else:
        visible_subject_mask = ImageChops.multiply(
            original_subject_mask.convert("L"),
            cut_component_mask.convert("L"),
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

    sticker_mask = nodes_module._expand_mask(
        cut_source_mask,
        border_size,
        merge_strength=merge_strength,
    )

    if border_softness > 0:
        sticker_mask = sticker_mask.filter(
            ImageFilter.GaussianBlur(radius=float(border_softness))
        )

    art, visible_subject_mask, sticker_mask = nodes_module._crop_with_padding(
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


def install_smart_sticker_filter(nodes_module) -> None:
    """Install v0.2.1 component-aware sticker extraction into nodes.py."""

    def patched_make_sticker(
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
    ):
        return _make_sticker_smart(
            nodes_module,
            image,
            removal_mode,
            white_threshold,
            edge_feather,
            border_size,
            border_softness,
            border_rgb,
            merge_strength,
            minimum_island_size,
            preserve_loose_splashes,
        )

    nodes_module._make_sticker = patched_make_sticker
