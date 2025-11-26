"""
siglip2_vit.py
"""

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported SigLIP Vision Backbones (from TIMM) =>> Note:: Using SigLIP w/ Patch = 14 (but SO400M Arch)
# timm.create_model() will add prefix "timm/"
SIGLIP2_VISION_BACKBONES = {
    "siglip2-vit-b16-224px": "ViT-B-16-SigLIP2",
    "siglip2-vit-b16-256px": "ViT-B-16-SigLIP2-256",
    "siglip2-vit-b16-384px": "ViT-B-16-SigLIP2-384",
    "siglip2-vit-b16-512px": "ViT-B-16-SigLIP2-512",

    "siglip2-vit-so400m": "ViT-SO400M-14-SigLIP2",      # SO  = sigmoid optimized, 400m parameters
    "siglip2-vit-so400m-384px": "ViT-SO400M-16-SigLIP2-384",
}


class SigLIP2ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            SIGLIP2_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
