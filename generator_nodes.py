import json
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def _tensor_batch_to_pil(image: torch.Tensor):
    tensor = image.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    images = []
    for frame in tensor:
        array = (frame.numpy() * 255.0 + 0.5).astype(np.uint8)
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def _background_rgb(background: str) -> Tuple[int, int, int]:
    return {
        "white": (255, 255, 255),
        "warm white": (248, 246, 241),
        "light grey": (238, 238, 238),
        "mid grey": (180, 180, 180),
        "black": (0, 0, 0),
    }.get(background, (255, 255, 255))


class OPSSquareImagePrep:
    """Normalise mixed source images to a centred square canvas without distortion."""

    SIZES = ("1024", "1536", "2048")
    FIT_MODES = ("contain", "cover")
    BACKGROUNDS = ("white", "warm white", "light grey", "mid grey", "black")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "square_size": (cls.SIZES, {"default": "1024"}),
                "fit_mode": (cls.FIT_MODES, {"default": "contain"}),
                "background": (cls.BACKGROUNDS, {"default": "white"}),
                "margin_percent": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 25.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("square_image", "square_info")
    FUNCTION = "prepare"
    CATEGORY = "OPS/Generation"

    def prepare(self, image, square_size, fit_mode, background, margin_percent):
        size = int(square_size)
        bg = _background_rgb(background)
        output = []

        for source in _tensor_batch_to_pil(image):
            if fit_mode == "cover":
                squared = ImageOps.fit(
                    source.convert("RGB"),
                    (size, size),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            else:
                margin = int(round(size * max(0.0, min(25.0, float(margin_percent))) / 100.0))
                inner = max(1, size - (margin * 2))
                fitted = ImageOps.contain(
                    source.convert("RGB"),
                    (inner, inner),
                    method=Image.Resampling.LANCZOS,
                )
                squared = Image.new("RGB", (size, size), bg)
                x = (size - fitted.width) // 2
                y = (size - fitted.height) // 2
                squared.paste(fitted, (x, y))
            output.append(_pil_to_tensor(squared))

        batch = torch.stack(output, dim=0)
        info = (
            f"{size}x{size} 1:1 square; {fit_mode}; centred; "
            f"background {background}; margin {float(margin_percent):.1f}%"
        )
        return (batch, info)


class OPSGeneratorConfig:
    """One place to choose the provider and record the preferred native settings for each API branch."""

    GENERATORS = ("OpenAI", "Grok", "Seedream")
    # 1024 and 2048 are the common square targets across the three current providers.
    SQUARE_SIZES = ("1024", "2048")

    OPENAI_MODELS = ("gpt-image-1", "gpt-image-1.5", "gpt-image-2")
    OPENAI_QUALITY = ("low", "medium", "high")

    GROK_MODELS = (
        "grok-imagine-image",
        "grok-imagine-image-quality",
        "grok-imagine-image-pro",
    )

    SEEDREAM_MODELS = (
        "seedream-4-0-250828",
        "seedream-4-5-251128",
        "seedream 5.0 lite",
        "seedream 5.0 pro",
    )
    SEEDREAM_PROMPT_OPTIMIZATION = ("standard", "disabled")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generator": (cls.GENERATORS, {"default": "Grok"}),
                "square_size": (cls.SQUARE_SIZES, {"default": "1024"}),
                "openai_model": (cls.OPENAI_MODELS, {"default": "gpt-image-1.5"}),
                "openai_quality": (cls.OPENAI_QUALITY, {"default": "low"}),
                "grok_model": (cls.GROK_MODELS, {"default": "grok-imagine-image"}),
                "seedream_model": (cls.SEEDREAM_MODELS, {"default": "seedream-4-0-250828"}),
                "seedream_prompt_optimization": (
                    cls.SEEDREAM_PROMPT_OPTIMIZATION,
                    {"default": "standard"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "generator",
        "selected_model",
        "square_size",
        "width",
        "height",
        "settings_json",
        "settings_summary",
    )
    FUNCTION = "build"
    CATEGORY = "OPS/Generation"

    def build(
        self,
        generator,
        square_size,
        openai_model,
        openai_quality,
        grok_model,
        seedream_model,
        seedream_prompt_optimization,
    ):
        size = int(square_size)
        grok_resolution = "1K" if size <= 1024 else "2K"

        if generator == "OpenAI":
            selected_model = openai_model
            provider_settings = {
                "provider": "OpenAI",
                "model": openai_model,
                "quality": openai_quality,
                "size": f"{size}x{size}",
                "aspect_ratio": "1:1",
            }
        elif generator == "Seedream":
            selected_model = seedream_model
            provider_settings = {
                "provider": "Seedream",
                "model": seedream_model,
                "prompt_optimization": seedream_prompt_optimization,
                "width": size,
                "height": size,
                "aspect_ratio": "1:1",
            }
        else:
            selected_model = grok_model
            provider_settings = {
                "provider": "Grok",
                "model": grok_model,
                "resolution": grok_resolution,
                "aspect_ratio": "1:1",
                "number_of_images": 1,
            }

        settings = {
            "generator": generator,
            "selected_model": selected_model,
            "square_size": size,
            "width": size,
            "height": size,
            "provider_settings": provider_settings,
            "openai": {"model": openai_model, "quality": openai_quality},
            "grok": {"model": grok_model, "resolution": grok_resolution},
            "seedream": {
                "model": seedream_model,
                "prompt_optimization": seedream_prompt_optimization,
            },
        }

        summary = f"{generator} / {selected_model} / {size}x{size} 1:1"
        if generator == "OpenAI":
            summary += f" / quality {openai_quality}"
        elif generator == "Grok":
            summary += f" / {grok_resolution}"
        else:
            summary += f" / prompt optimisation {seedream_prompt_optimization}"

        return (
            generator,
            selected_model,
            size,
            size,
            size,
            json.dumps(settings, ensure_ascii=False),
            summary,
        )


class OPSGeneratorRouter:
    """Lazy image router: only requests the API branch selected by OPS Generator Config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Feed this from OPS Generator Config so one dropdown controls every router in the workflow.
                "generator": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "openai_image": ("IMAGE", {"lazy": True}),
                "grok_image": ("IMAGE", {"lazy": True}),
                "seedream_image": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_image", "selected_generator")
    FUNCTION = "route"
    CATEGORY = "OPS/Generation"

    @staticmethod
    def _normalise_generator(generator: str) -> str:
        value = str(generator or "Grok").strip().lower()
        if value == "openai":
            return "OpenAI"
        if value == "seedream":
            return "Seedream"
        return "Grok"

    @classmethod
    def _input_for(cls, generator: str) -> str:
        generator = cls._normalise_generator(generator)
        return {
            "OpenAI": "openai_image",
            "Grok": "grok_image",
            "Seedream": "seedream_image",
        }[generator]

    def check_lazy_status(
        self,
        generator,
        openai_image=None,
        grok_image=None,
        seedream_image=None,
    ):
        input_name = self._input_for(generator)
        current = {
            "openai_image": openai_image,
            "grok_image": grok_image,
            "seedream_image": seedream_image,
        }[input_name]
        return [input_name] if current is None else []

    def route(
        self,
        generator,
        openai_image=None,
        grok_image=None,
        seedream_image=None,
    ):
        generator = self._normalise_generator(generator)
        selected = {
            "OpenAI": openai_image,
            "Grok": grok_image,
            "Seedream": seedream_image,
        }[generator]

        if selected is None:
            selected = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (selected, generator)


NODE_CLASS_MAPPINGS = {
    "OPSSquareImagePrep": OPSSquareImagePrep,
    "OPSGeneratorConfig": OPSGeneratorConfig,
    "OPSGeneratorRouter": OPSGeneratorRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPSSquareImagePrep": "OPS Square Image Prep",
    "OPSGeneratorConfig": "OPS Generator Config",
    "OPSGeneratorRouter": "OPS Generator Router",
}
