import os
import timm
import torch.nn as nn
from torchvision.models import resnet18

if os.environ["CONDA_PREFIX"] == "/Users/mblons/miniforge3/envs/conch-env":
    from conch.open_clip_custom import create_model_from_pretrained
if os.environ["CONDA_PREFIX"] == "/Users/mblons/miniforge3/envs/ctranspath-env":
    from timm.models.layers.helpers import to_2tuple


####################################################################################
# The following lines of code were fork as is from:
# https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py
####################################################################################
class ConvStem(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False
    )
    return model


def uni():
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    return model


def conch(model_path):
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", model_path, force_image_size=224
    )
    return model, preprocess


def moco():
    model = resnet18()
    return model


def imagenet():
    model = resnet18(weights="IMAGENET1K_V1")
    return model


def gigapath():
    if not "HF_TOKEN" in os.environ:
        model = timm.create_model(
            "vit_giant_patch14_dinov2",
            img_size=224,
            in_chans=3,
            patch_size=16,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            init_values=1e-05,
            mlp_ratio=5.33334,
            num_classes=0,
        )
    else:
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    return model
