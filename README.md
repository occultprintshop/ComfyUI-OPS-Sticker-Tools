# ComfyUI OPS Sticker Tools

A small ComfyUI custom-node pack for creating sticker-style featured images and mockups.

## Node: OPS Featured Sticker Mockup

The first node is deliberately focused on the common product-image workflow:

1. load artwork on a white background;
2. remove only the outer white background (or all white if desired);
3. add a smooth white sticker border;
4. add an optional soft shadow;
5. place the finished sticker onto a background image or simple canvas.

### Outputs

- **mockup** — finished featured/mockup image.
- **sticker_rgb** — sticker artwork on black outside the cut shape.
- **sticker_mask** — alpha/cut mask. Combine this with `sticker_rgb` in ComfyUI when you need transparency.

### Useful starting settings

- Remove background: `edge white`
- White threshold: `245`
- Edge feather: `1.2`
- Border size: `22`
- Border softness: `0.6`
- Sticker size: `72%`
- X / Y: `50 / 50`
- Shadow opacity: `0.20`
- Shadow blur: `22`

## Install

Clone the repository into your ComfyUI custom nodes folder, then restart ComfyUI.

```bash
cd /Users/macbookpro/Documents/ComfyUI/custom_nodes
git clone https://github.com/occultprintshop/ComfyUI-OPS-Sticker-Tools.git
```

The node appears under:

`OPS > Mockups > OPS Featured Sticker Mockup`

## Dependencies

No extra packages beyond normal ComfyUI dependencies: PyTorch, NumPy and Pillow.
