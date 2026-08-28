# ComfyUI OPS Sticker Tools

A lightweight ComfyUI custom-node pack for OPS sticker extraction, deterministic product mockups and apparel mockup workflows.

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

## v0.3.0 Apparel Mockup Pipeline

New nodes appear under `OPS > Apparel`.

### OPS Model Library

Selects one of the fixed OPS base-model profiles and outputs:

- the model reference image;
- model ID;
- human-readable profile summary;
- profile JSON for downstream prompt building.

The first roster contains eight adult profiles across XS/S, M/L, XL/2XL and 3XL/5XL body-size reference groups for male and female models.

Approved model files can be placed under:

`assets/models/<MODEL_ID>/front.webp`

For example:

`assets/models/M_ML_01/front.webp`

Until the approved images are added, the node produces a labelled placeholder. A normal ComfyUI `IMAGE` can also be connected to **reference_override** for testing with any supplied model image while keeping the selected profile metadata.

### OPS Apparel Setup

Builds the garment and scene recipe including:

- garment type and product name;
- XS–5XL garment size;
- colour and fit;
- pose;
- camera angle;
- occasion/context;
- background description.

The selected model body-size reference and the actual garment size remain separate values.

### OPS Design Setup

Passes the original artwork through untouched while storing its intended apparel placement:

- centre/full front;
- left chest;
- centre/full back;
- sleeves;
- print scale;
- X/Y placement offset.

### OPS Grok Prompt Builder

Combines the selected model profile, apparel recipe and design recipe into:

- a single reference-board image containing the model and exact artwork;
- a structured Grok edit/generation prompt;
- the complete recipe JSON.

The prompt tells the image generator to preserve model identity, apparent age, body proportions and size reference while creating realistic fabric, folds, seams, shadows and print interaction. The original print artwork should still be reapplied in a deterministic finishing pass when exact reproduction is required.

### OPS Approval Gate

A lazy two-stage switch:

- `PROOF_ONLY` requests only the proof-image branch;
- `APPROVED` requests the full-mockup branch.

The unused image branch is declared lazy so expensive downstream API generation does not need to run while the proof is still being reviewed.

### OPS Mockup Shot Planner

Turns an approved base prompt into a list of continuity-preserving shot prompts.

Shot packs:

- **essential** — front, back and two three-quarter views;
- **standard** — adds left/right profile and seated;
- **full** — adds kneeling, walking away, high/low angle and detail views.

The planner tells downstream generation to preserve the same model, garment, size, colour, fit, artwork and scene styling across the full set.

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

The v0.3 apparel workflow will be added after the new apparel nodes have been pulled, loaded and tested in ComfyUI.

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
python3 -m py_compile __init__.py nodes.py smart_sticker.py apparel_nodes.py
```

Then restart ComfyUI.

Nodes appear under:

- `OPS > Mockups`
- `OPS > Apparel`

## Dependencies

No additional Python packages are required beyond the normal ComfyUI environment: PyTorch, NumPy and Pillow.
