"""
dinov3_vit.py
"""

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported DINOv3 Vision Backbones (from TIMM) =>> Note:: Using DINOv3 w/ Registers!
DINOv3_VISION_BACKBONES = {"dinov3-vit-l": "vit_large_patch16_dinov3.lvd1689m"} # patch=1 => 224/16 = 14 patches


class DinoV3ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            DINOv3_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
