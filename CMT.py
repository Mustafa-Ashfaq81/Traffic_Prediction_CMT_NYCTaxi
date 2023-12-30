import torch
import torch.nn as nn
from cmt_module import CMTStem, Patch_Aggregate, CMTBlock
from Param_Our import *
from typing import List, Optional
from tqdm import tqdm

class CMT(nn.Module):
    def __init__(self,
        in_channels = 3,
        stem_channel = 32,
        cmt_channel = [46, 92, 184, 368],
        patch_channel = [46, 92, 184, 368],
        block_layer = [2, 2, 10, 2],
        R = 3.6,
        img_size = 224,
        num_class = 10
    ):
        super(CMT, self).__init__()

        # Image size for each stage
        size = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        # Stem layer
        self.stem = CMTStem(in_channels, stem_channel)

        # Patch Aggregation Layer
        self.patch1 = Patch_Aggregate(stem_channel, patch_channel[0])
        self.patch2 = Patch_Aggregate(patch_channel[0], patch_channel[1])
        self.patch3 = Patch_Aggregate(patch_channel[1], patch_channel[2])
        self.patch4 = Patch_Aggregate(patch_channel[2], patch_channel[3])

        # CMT Block Layer
        stage1 = []
        for _ in range(block_layer[0]):
            cmt_layer = CMTBlock(
                    img_size = size[0],
                    stride = 8,
                    d_k = cmt_channel[0],
                    d_v = cmt_channel[0],
                    num_heads = 1,
                    R = R,
                    in_channels = patch_channel[0]
            )
            stage1.append(cmt_layer)
        self.stage1 = nn.Sequential(*stage1)

        stage2 = []
        for _ in range(block_layer[1]):
            cmt_layer = CMTBlock(
                    img_size = size[1],
                stride = 4,
                d_k = cmt_channel[1] // 2,
                d_v = cmt_channel[1] // 2,
                num_heads = 2,
                R = R,
                in_channels = patch_channel[1]
            )
            stage2.append(cmt_layer)
        self.stage2 = nn.Sequential(*stage2)

        stage3 = []
        for _ in range(block_layer[2]):
            cmt_layer = CMTBlock(
                img_size = size[2],
                stride = 2,
                d_k = cmt_channel[2] // 4,
                d_v = cmt_channel[2] // 4,
                num_heads = 4,
                R = R,
                in_channels = patch_channel[2]
            )
            stage3.append(cmt_layer)
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        for _ in range(block_layer[3]):
            cmt_layer = CMTBlock(
                img_size = size[3],
                stride = 1,
                d_k = cmt_channel[3] // 8,
                d_v = cmt_channel[3] // 8,
                num_heads = 8,
                R = R,
                in_channels = patch_channel[3]
            )
            stage4.append(cmt_layer)
        self.stage4 = nn.Sequential(*stage4)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(368//2, 1280),
            # nn.ReLU(inplace = True),
        )

        # Final Regression Layer
        self.regression = nn.Linear(1280, Prediction_TIMESTEP * local_image_size_x * local_image_size_y)




    def forward(self, x):
        # print(f"\nShape before STEM: {x.shape}")
        x = self.stem(x)
        # print(f"Shape after STEM: {x.shape}")
        # print()

        # print(f"Shape before Patch-Aggregation 1: {x.shape}")
        x = self.patch1(x)
        # print(f"Shape after Patch-Aggregation 1: {x.shape}")
        # print()
        # print(f"Shape before Block 1: {x.shape}")
        x = self.stage1(x)
        # print(f"Shape after Block 1: {x.shape}")

        # print(f"Shape before Patch-Aggregation 2: {x.shape}")
        x = self.patch2(x)
        # print(f"Shape after Patch-Aggregation 2: {x.shape}")
        # print()

        # print(f"Shape before Block 2: {x.shape}")
        x = self.stage2(x)
        # print(f"Shape after Block 2: {x.shape}")

        # print(f"Shape before Patch-Aggregation 3: {x.shape}")
        x = self.patch3(x)
        # print(f"Shape after Patch-Aggregation 3: {x.shape}")
        # print()

        # print(f"Shape before Block 3: {x.shape}")
        x = self.stage3(x)
        # print(f"Shape after Block 3: {x.shape}")

        # print(f"Shape before Patch-Aggregation 4: {x.shape}")
        x = self.patch4(x)
        # print(f"Shape after Patch-Aggregation 4: {x.shape}")
        # print()

        # print(f"Shape before Block 4: {x.shape}")
        x = self.stage4(x)
        # print(f"Shape after Block 4: {x.shape}")
        
        # print(f"Shape before avg pool: {x.shape}")
        x = self.avg_pool(x)
        # print(f"Shape after avg pool: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"Shape after flatten: {x.shape}")

        x = self.fc(x)
        # print(f"Shape after fc: {x.shape}")
        # print()
        logit = self.regression(x)
        # print("Shape after final regression:", logit.shape)

        size_for_reshape = Prediction_TIMESTEP * local_image_size_x * local_image_size_y
        logit = logit.view(-1, size_for_reshape)
        # print(f"Shape after reshape: {logit.shape}")

        # Apply Linear transformation equivalent to Dense layer
        linear = nn.Linear(size_for_reshape, Prediction_TIMESTEP * local_image_size_x * local_image_size_y)(logit)
        # print(f"Shape after Linear/Dense layer: {linear.shape}")

        # Reshape the tensor
        res = linear.view(-1, Prediction_TIMESTEP, local_image_size_x, local_image_size_y)
        # print(f"Final Reshape: {res.shape}")

        # Add a new dimension
        res = res.unsqueeze(1)
        return res


def CMT_Ti(img_size = 224, num_class = 1000):
    model = CMT(
        in_channels = 4,
        stem_channel = 16//2,
        cmt_channel = [46//2, 92//2, 184//2, 368//2],
        patch_channel = [46//2, 92//2, 184//2, 368//2],
        block_layer = [2, 2, 10//2, 2],
        R = 3.6,
        img_size = img_size,
        num_class = num_class
    )
    return model


def CMT_XS(img_size = 224, num_class = 1000):
    model = CMT(
        in_channels = 4,
        stem_channel = 16,
        cmt_channel = [52, 104, 208, 416],
        patch_channel = [52, 104, 208, 416],
        block_layer = [3, 3, 12, 3],
        R = 3.8,
        img_size = img_size,
        num_class = num_class
    )
    return model

def CMT_S(img_size = 224, num_class = 10):
    model = CMT(
        in_channels = 4,
        stem_channel = 32,
        cmt_channel = [64, 128, 256, 512],
        patch_channel = [64, 128, 256, 512],
        block_layer = [3, 3, 16, 3],
        R = 4,
        img_size = img_size,
        num_class = num_class
    )
    return model

def CMT_B(img_size = 224, num_class = 10):
    model = CMT(
        in_channels = 4,
        stem_channel = 38,
        cmt_channel = [76, 152, 304, 608],
        patch_channel = [76, 152, 304, 608],
        block_layer = [4, 4, 20, 4],
        R = 4,
        img_size = img_size,
        num_class = num_class
    )
    return model


def test():
    calc_param = lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)
    img = torch.rand(2, 3, 224, 224)
    cmt_ti = CMT_Ti()
    cmt_xs = CMT_XS()
    cmt_x = CMT_S()
    cmt_b = CMT_B()
    logit = cmt_b(img)
    print(logit.size())
    print(f"CMT_Ti param: {calc_param(cmt_ti) / 1e6 : .2f} M")
    print(f"CMT_XS param: {calc_param(cmt_xs) / 1e6 : .2f} M")
    print(f"CMT_X  param: {calc_param(cmt_x) / 1e6 : .2f} M")
    print(f"CMT_B  param: {calc_param(cmt_b) / 1e6 : .2f} M")

# if __name__ == "__main__":
#     x = torch.randn(2, 4, 30, 30) #(B,C,H,W)
#     # print(x)
#     model = CMT_S()

#     # print(model)
#     print("---------------------------")
#     print()
#     out = model(x)
#     print(f"Final shape of output: {out.shape}")