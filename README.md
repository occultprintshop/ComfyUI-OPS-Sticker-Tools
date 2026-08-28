# ComfyUI OPS Sticker Tools

A lightweight ComfyUI custom-node pack for OPS sticker extraction and deterministic product mockups.

## Nodes

### OPS Featured Sticker Mockup

Takes artwork on a white background and:

1. removes the outer white background;
2. builds a rounded sticker cut silhouette;
3. adds a white border and optional shadow;
4. places the result on a background or simple canvas.

Outputs:

- **mockup** — finished featured/mockup image.
- **sticker_rgb** — sticker artwork on black outside the cut shape.
- **sticker_mask** — physical cut/shadow mask.

Useful starting settings:

- Remove background: `edge white`
- White threshold: `245`
- Edge feather: `1.2`
- Border size: `22`
- Border softness: `0.6`
- Sticker size: `72%`
- Shadow opacity: `0.20`
- Shadow blur: `22`

#### Watercolour controls

These are optional so older saved workflows still run.

- **merge_strength** — increases or reduces how strongly nearby paint flecks are joined into one cut silhouette. `0` disables the extra merge pass; `1` matches the v0.1.3 behaviour.
- **minimum_island_size** — approximate small-island diameter in pixels to ignore when building the physical cut silhouette. Leave at `0` for normal artwork.
- **preserve_loose_splashes** — when enabled, loose watercolour marks can remain visible even if they are excluded from the physical cut silhouette.

For pale watercolour artwork, remember that a **higher** `white_threshold` is less aggressive. Try `250–253` with a lower `edge_feather` if faint beige/grey detail is being lost.

### OPS Product Template Mockup

Places a complete design into a supplied mockup template and mask **without removing the design's white background**.

This is intended for:

- greeting cards;
- 30 mm / 55 mm badges;
- T-shirts and hoodies;
- tote bags;
- posters and other masked product templates.

Inputs include:

- artwork;
- template image;
- mask image;
- `cover`, `contain` or `stretch` fitting;
- scale, X/Y offset and rotation;
- mask inset;
- edge blending;
- optional template shading preservation.

Outputs:

- **mockup** — completed product mockup.
- **fitted_artwork** — the full-size fitted artwork layer.
- **effective_mask** — the final mask used for placement.

The supplied mask image can be connected directly from the normal `IMAGE` output of a ComfyUI `LoadImage` node. White represents the product placement area by default.

## Workflows

`workflows/OPS_Featured_Sticker_Mockup_v0.2_API.json`

Updated sticker workflow including the optional watercolour controls.

`workflows/OPS_Product_Template_Mockup_API.json`

Starter three-way product workflow using these filenames:

- `YOUR_ARTWORK.png`
- `CARD MOCK UP.png`
- `CARD MOCK UP MASK.png`
- `Small Badge.png`
- `Small Badge Mask.png`
- `Large Badge.png`
- `Large Badge Mask.png`

It renders the same artwork to the greeting card, 30 mm badge and 55 mm badge templates.

## Install / update

Clone into the ComfyUI custom nodes directory:

```bash
cd /Users/macbookpro/Documents/ComfyUI/custom_nodes
git clone https://github.com/occultprintshop/ComfyUI-OPS-Sticker-Tools.git
```

For an existing install:

```bash
cd /Users/macbookpro/Documents/ComfyUI/custom_nodes/ComfyUI-OPS-Sticker-Tools
git pull
python3 -m py_compile __init__.py nodes.py
```

Then restart ComfyUI.

Nodes appear under:

`OPS > Mockups`

## Dependencies

No additional Python packages are required beyond the normal ComfyUI environment: PyTorch, NumPy and Pillow.
