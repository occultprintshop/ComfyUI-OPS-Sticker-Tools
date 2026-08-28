import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_MANIFEST_PATH = PACKAGE_ROOT / "assets" / "models" / "model_profiles.json"

FALLBACK_PROFILES = {
    "M_XSS_01": {
        "sex": "male",
        "size_band": "XS/S",
        "age": 27,
        "appearance": "White / European",
        "hair": "short brown hair",
        "build": "lean frame, narrower shoulders",
        "height_impression": "average",
        "base_layer": "plain neutral fitted athletic top and shorts",
        "image": "M_XSS_01/front.webp",
    },
    "M_ML_01": {
        "sex": "male",
        "size_band": "M/L",
        "age": 31,
        "appearance": "South Asian / Indian",
        "hair": "short black slightly wavy hair",
        "build": "average balanced frame",
        "height_impression": "slightly tall",
        "base_layer": "plain neutral fitted athletic top and shorts",
        "image": "M_ML_01/front.webp",
    },
    "M_XL2XL_01": {
        "sex": "male",
        "size_band": "XL/2XL",
        "age": 38,
        "appearance": "East Asian / Chinese",
        "hair": "short dark hair",
        "build": "broad chest, solid frame",
        "height_impression": "average",
        "base_layer": "plain neutral fitted athletic top and shorts",
        "image": "M_XL2XL_01/front.webp",
    },
    "M_3XL5XL_01": {
        "sex": "male",
        "size_band": "3XL/5XL",
        "age": 42,
        "appearance": "Black / African diaspora",
        "hair": "close-cropped black hair",
        "build": "larger broad frame",
        "height_impression": "tall",
        "base_layer": "plain neutral fitted athletic top and shorts",
        "image": "M_3XL5XL_01/front.webp",
    },
    "F_XSS_01": {
        "sex": "female",
        "size_band": "XS/S",
        "age": 28,
        "appearance": "East Asian / Chinese",
        "hair": "shoulder-length dark hair",
        "build": "petite slim frame",
        "height_impression": "shorter",
        "base_layer": "plain neutral fitted athletic top and shorts or leggings",
        "image": "F_XSS_01/front.webp",
    },
    "F_ML_01": {
        "sex": "female",
        "size_band": "M/L",
        "age": 33,
        "appearance": "Black / African diaspora",
        "hair": "natural curly dark hair",
        "build": "average balanced frame",
        "height_impression": "average",
        "base_layer": "plain neutral fitted athletic top and shorts or leggings",
        "image": "F_ML_01/front.webp",
    },
    "F_XL2XL_01": {
        "sex": "female",
        "size_band": "XL/2XL",
        "age": 39,
        "appearance": "White / European",
        "hair": "medium-length auburn hair",
        "build": "fuller broad-balanced frame",
        "height_impression": "slightly tall",
        "base_layer": "plain neutral fitted athletic top and shorts or leggings",
        "image": "F_XL2XL_01/front.webp",
    },
    "F_3XL5XL_01": {
        "sex": "female",
        "size_band": "3XL/5XL",
        "age": 43,
        "appearance": "South Asian / Indian",
        "hair": "long dark hair",
        "build": "larger full frame",
        "height_impression": "average",
        "base_layer": "plain neutral fitted athletic top and shorts or leggings",
        "image": "F_3XL5XL_01/front.webp",
    },
}


def _load_profiles() -> Dict[str, dict]:
    if MODEL_MANIFEST_PATH.exists():
        try:
            data = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
            profiles = data.get("profiles", data)
            if isinstance(profiles, dict) and profiles:
                return profiles
        except Exception:
            pass
    return FALLBACK_PROFILES


MODEL_PROFILES = _load_profiles()
MODEL_IDS = list(MODEL_PROFILES.keys())


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    tensor = image[0].detach().cpu().float().clamp(0.0, 1.0)
    array = (tensor.numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _contain_on_canvas(image: Image.Image, size: Tuple[int, int], background=(245, 245, 245)) -> Image.Image:
    canvas = Image.new("RGB", size, background)
    fitted = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _profile_summary(model_id: str, profile: dict) -> str:
    return (
        f"{model_id}: {profile.get('sex', 'adult')}, size reference {profile.get('size_band', '')}, "
        f"age {profile.get('age', '')}, {profile.get('appearance', '')}; "
        f"{profile.get('build', '')}; {profile.get('height_impression', '')} height impression; "
        f"{profile.get('hair', '')}."
    )


def _placeholder_model(model_id: str, profile: dict) -> Image.Image:
    width, height = 768, 1024
    image = Image.new("RGB", (width, height), (242, 242, 242))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 45, width - 45, height - 45), radius=28, outline=(115, 115, 115), width=3)

    cx = width // 2
    head_y = 250
    draw.ellipse((cx - 65, head_y - 65, cx + 65, head_y + 65), fill=(195, 195, 195))
    draw.rounded_rectangle((cx - 120, 325, cx + 120, 700), radius=80, fill=(205, 205, 205))
    draw.rounded_rectangle((cx - 175, 355, cx - 105, 690), radius=35, fill=(205, 205, 205))
    draw.rounded_rectangle((cx + 105, 355, cx + 175, 690), radius=35, fill=(205, 205, 205))
    draw.rounded_rectangle((cx - 105, 650, cx - 20, 920), radius=35, fill=(200, 200, 200))
    draw.rounded_rectangle((cx + 20, 650, cx + 105, 920), radius=35, fill=(200, 200, 200))

    draw.rectangle((0, 0, width, 130), fill=(32, 32, 32))
    draw.text((36, 38), f"OPS MODEL  {model_id}", fill=(255, 255, 255))
    summary = _profile_summary(model_id, profile)
    draw.rectangle((0, 934, width, height), fill=(255, 255, 255))
    draw.text((25, 955), summary[:105], fill=(30, 30, 30))
    return image


def _load_model_asset(model_id: str, profile: dict) -> Image.Image:
    relative = profile.get("image", f"{model_id}/front.webp")
    path = PACKAGE_ROOT / "assets" / "models" / relative
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return _placeholder_model(model_id, profile)


def _parse_json(text: str) -> dict:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


class OPSModelLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (MODEL_IDS,),
            },
            "optional": {
                "reference_override": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_reference", "model_id", "profile_summary", "profile_json")
    FUNCTION = "load_model"
    CATEGORY = "OPS/Apparel"

    def load_model(self, model_id, reference_override=None):
        profile = MODEL_PROFILES.get(model_id, FALLBACK_PROFILES[MODEL_IDS[0]])
        if reference_override is not None:
            image = reference_override
        else:
            image = _pil_to_tensor(_load_model_asset(model_id, profile))
        summary = _profile_summary(model_id, profile)
        profile_json = json.dumps({"model_id": model_id, **profile}, ensure_ascii=False)
        return (image, model_id, summary, profile_json)


class OPSApparelSetup:
    GARMENTS = (
        "heavyweight crewneck t-shirt",
        "premium crewneck t-shirt",
        "tank top",
        "pullover hoodie",
        "zip hoodie",
        "crewneck sweatshirt",
        "joggers",
        "custom / other",
    )
    SIZES = ("XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL")
    FITS = ("regular", "relaxed", "oversized", "fitted")
    POSES = (
        "front standing",
        "back standing",
        "left side",
        "right side",
        "3/4 front left",
        "3/4 front right",
        "seated",
        "kneeling",
        "walking toward camera",
        "walking away",
        "hands in pockets",
        "arms folded",
    )
    CAMERA_ANGLES = ("eye level", "slightly above", "slightly below", "high angle", "low angle")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "garment_type": (cls.GARMENTS,),
                "product_name": ("STRING", {"default": "Gildan 5000", "multiline": False}),
                "garment_size": (cls.SIZES,),
                "colour": ("STRING", {"default": "white", "multiline": False}),
                "fit": (cls.FITS,),
                "pose": (cls.POSES,),
                "camera_angle": (cls.CAMERA_ANGLES,),
                "occasion": ("STRING", {"default": "clean ecommerce studio", "multiline": False}),
                "background": ("STRING", {"default": "soft neutral studio background", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("apparel_recipe", "apparel_summary")
    FUNCTION = "build"
    CATEGORY = "OPS/Apparel"

    def build(self, garment_type, product_name, garment_size, colour, fit, pose, camera_angle, occasion, background):
        recipe = {
            "garment_type": garment_type,
            "product_name": product_name.strip(),
            "garment_size": garment_size,
            "colour": colour.strip(),
            "fit": fit,
            "pose": pose,
            "camera_angle": camera_angle,
            "occasion": occasion.strip(),
            "background": background.strip(),
        }
        summary = (
            f"{recipe['product_name'] or garment_type}, {garment_size}, {recipe['colour']}, {fit} fit; "
            f"{pose}; {camera_angle}; {recipe['occasion']}; background: {recipe['background']}."
        )
        return (json.dumps(recipe, ensure_ascii=False), summary)


class OPSDesignSetup:
    PLACEMENTS = (
        "centre front",
        "full front",
        "left chest",
        "centre back",
        "full back",
        "left sleeve",
        "right sleeve",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "design": ("IMAGE",),
                "placement": (cls.PLACEMENTS,),
                "print_scale_percent": ("FLOAT", {"default": 72.0, "min": 5.0, "max": 150.0, "step": 1.0}),
                "x_offset_percent": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0}),
                "y_offset_percent": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0}),
                "design_background": (("transparent / isolated", "keep original background"),),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("design", "design_recipe", "design_summary")
    FUNCTION = "prepare"
    CATEGORY = "OPS/Apparel"

    def prepare(self, design, placement, print_scale_percent, x_offset_percent, y_offset_percent, design_background):
        recipe = {
            "placement": placement,
            "print_scale_percent": float(print_scale_percent),
            "x_offset_percent": float(x_offset_percent),
            "y_offset_percent": float(y_offset_percent),
            "design_background": design_background,
        }
        summary = (
            f"{placement}, scale {print_scale_percent:.0f}%, "
            f"X {x_offset_percent:+.0f}%, Y {y_offset_percent:+.0f}%, {design_background}."
        )
        return (design, json.dumps(recipe, ensure_ascii=False), summary)


class OPSGrokPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_reference": ("IMAGE",),
                "design": ("IMAGE",),
                "profile_json": ("STRING", {"multiline": True, "default": "{}"}),
                "apparel_recipe": ("STRING", {"multiline": True, "default": "{}"}),
                "design_recipe": ("STRING", {"multiline": True, "default": "{}"}),
                "extra_instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Keep the result photorealistic, commercially usable and consistent with the supplied references.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("reference_board", "grok_prompt", "recipe_json")
    FUNCTION = "build"
    CATEGORY = "OPS/Apparel"

    def build(self, model_reference, design, profile_json, apparel_recipe, design_recipe, extra_instructions):
        profile = _parse_json(profile_json)
        apparel = _parse_json(apparel_recipe)
        design_data = _parse_json(design_recipe)

        model_id = profile.get("model_id", "selected OPS model")
        prompt = (
            f"Create a polished apparel product mockup using the supplied reference board. "
            f"Preserve the identity, age impression, facial features, skin tone, hair, body proportions, "
            f"body-size reference and overall scale of model {model_id}. "
            f"The model should wear {apparel.get('product_name') or apparel.get('garment_type', 'the specified garment')} "
            f"in size {apparel.get('garment_size', '')}, colour {apparel.get('colour', '')}, "
            f"with a {apparel.get('fit', 'regular')} fit. "
            f"Pose: {apparel.get('pose', 'front standing')}. "
            f"Camera: {apparel.get('camera_angle', 'eye level')}. "
            f"Context/occasion: {apparel.get('occasion', 'clean ecommerce studio')}. "
            f"Background: {apparel.get('background', 'neutral studio')}. "
            f"Place the supplied print artwork at {design_data.get('placement', 'centre front')} "
            f"at approximately {design_data.get('print_scale_percent', 72)} percent scale, "
            f"offset X {design_data.get('x_offset_percent', 0)} percent and "
            f"Y {design_data.get('y_offset_percent', 0)} percent. "
            f"Treat the supplied artwork as a strict visual reference: do not redesign, rewrite, simplify, "
            f"replace or invent elements of the artwork. Make the garment itself realistic with believable fabric "
            f"drape, seams, folds, perspective, shadows and print interaction. "
            f"Do not change the selected model's apparent body size merely to fit the clothing size. "
            f"If exact artwork reproduction conflicts with fabric realism, prioritise a believable garment surface; "
            f"the original artwork will be reapplied in a deterministic finishing pass. "
            f"{extra_instructions.strip()}"
        )

        model_pil = _tensor_to_pil(model_reference)
        design_pil = _tensor_to_pil(design)

        board = Image.new("RGB", (1536, 1024), (235, 235, 235))
        draw = ImageDraw.Draw(board)
        draw.rectangle((0, 0, 1536, 80), fill=(28, 28, 28))
        draw.text((28, 28), f"OPS APPAREL REFERENCE — {model_id}", fill=(255, 255, 255))

        left = _contain_on_canvas(model_pil, (900, 880), (248, 248, 248))
        right = _contain_on_canvas(design_pil, (560, 650), (255, 255, 255))
        board.paste(left, (30, 110))
        board.paste(right, (940, 170))
        draw.text((940, 130), "EXACT PRINT ARTWORK", fill=(30, 30, 30))
        draw.text((940, 845), f"Placement: {design_data.get('placement', 'centre front')}", fill=(30, 30, 30))
        draw.text((940, 875), f"Scale: {design_data.get('print_scale_percent', 72)}%", fill=(30, 30, 30))
        draw.text((940, 905), f"Garment: {apparel.get('garment_size', '')} {apparel.get('colour', '')}", fill=(30, 30, 30))

        recipe = {
            "model": profile,
            "apparel": apparel,
            "design": design_data,
            "prompt": prompt,
        }
        return (_pil_to_tensor(board), prompt, json.dumps(recipe, ensure_ascii=False))


class OPSApprovalGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (("PROOF_ONLY", "APPROVED"),),
            },
            "optional": {
                "proof_image": ("IMAGE", {"lazy": True}),
                "full_mockup_image": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_image", "status")
    FUNCTION = "route"
    CATEGORY = "OPS/Apparel"

    def check_lazy_status(self, mode, proof_image=None, full_mockup_image=None):
        if mode == "PROOF_ONLY" and proof_image is None:
            return ["proof_image"]
        if mode == "APPROVED" and full_mockup_image is None:
            return ["full_mockup_image"]
        return []

    def route(self, mode, proof_image=None, full_mockup_image=None):
        selected = proof_image if mode == "PROOF_ONLY" else full_mockup_image
        if selected is None:
            selected = _pil_to_tensor(Image.new("RGB", (512, 512), (240, 240, 240)))
        status = "Proof only — full mockup branch is held." if mode == "PROOF_ONLY" else "Approved — full mockup branch enabled."
        return (selected, status)


class OPSMockupShotPlanner:
    SHOT_PACKS = ("essential", "standard", "full")

    SHOTS = {
        "essential": [
            ("front", "full-body front view, eye-level camera, relaxed natural stance"),
            ("back", "full-body back view, eye-level camera, relaxed natural stance"),
            ("three_quarter_left", "full-body three-quarter front-left view, eye-level camera"),
            ("three_quarter_right", "full-body three-quarter front-right view, eye-level camera"),
        ],
        "standard": [
            ("front", "full-body front view, eye-level camera, relaxed natural stance"),
            ("back", "full-body back view, eye-level camera, relaxed natural stance"),
            ("left_side", "full-body left profile view"),
            ("right_side", "full-body right profile view"),
            ("three_quarter_left", "full-body three-quarter front-left view"),
            ("three_quarter_right", "full-body three-quarter front-right view"),
            ("seated", "natural seated pose with garment print area still readable"),
        ],
        "full": [
            ("front", "full-body front view, eye-level camera, relaxed natural stance"),
            ("back", "full-body back view, eye-level camera, relaxed natural stance"),
            ("left_side", "full-body left profile view"),
            ("right_side", "full-body right profile view"),
            ("three_quarter_left", "full-body three-quarter front-left view"),
            ("three_quarter_right", "full-body three-quarter front-right view"),
            ("seated", "natural seated pose with realistic garment folds"),
            ("kneeling", "natural kneeling pose, commercially styled and non-dramatic"),
            ("walking_away", "walking away from camera, back of garment clearly visible"),
            ("high_angle", "full-body shot from a modestly elevated camera angle"),
            ("low_angle", "full-body shot from a modest low camera angle without distortion"),
            ("detail", "closer garment detail shot showing print, fabric texture and seams"),
        ],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "approved_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "shot_pack": (cls.SHOT_PACKS,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shot_prompts", "shot_names")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "plan"
    CATEGORY = "OPS/Apparel"

    def plan(self, approved_base_prompt, shot_pack):
        prompts: List[str] = []
        names: List[str] = []
        base = approved_base_prompt.strip()
        for name, instruction in self.SHOTS[shot_pack]:
            names.append(name)
            prompts.append(
                f"{base}\n\nAPPROVED MOCKUP CONTINUITY RULE: Keep the same model identity, apparent age, "
                f"body proportions, garment product, garment size, colour, fit, artwork and overall scene styling "
                f"from the approved proof. Change only the pose/camera composition required for this shot. "
                f"SHOT: {instruction}. Maintain exact visual continuity across the set."
            )
        return (prompts, names)


NODE_CLASS_MAPPINGS = {
    "OPSModelLibrary": OPSModelLibrary,
    "OPSApparelSetup": OPSApparelSetup,
    "OPSDesignSetup": OPSDesignSetup,
    "OPSGrokPromptBuilder": OPSGrokPromptBuilder,
    "OPSApprovalGate": OPSApprovalGate,
    "OPSMockupShotPlanner": OPSMockupShotPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPSModelLibrary": "OPS Model Library",
    "OPSApparelSetup": "OPS Apparel Setup",
    "OPSDesignSetup": "OPS Design Setup",
    "OPSGrokPromptBuilder": "OPS Grok Prompt Builder",
    "OPSApprovalGate": "OPS Approval Gate",
    "OPSMockupShotPlanner": "OPS Mockup Shot Planner",
}
