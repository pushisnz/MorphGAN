import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))
def depthwise_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    )
class SPADE(nn.Module):
    def __init__(self, norm_channels, cond_channels, ks=3, padding=1):

        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False) # 使用 InstanceNorm，不带学习的 affine 参数


        self.conv_gamma = conv2d(cond_channels, norm_channels, kernel_size=ks, padding=padding)

        self.conv_beta = conv2d(cond_channels, norm_channels, kernel_size=ks, padding=padding)

    def forward(self, x, condition):


        norm_x = self.norm(x)



        H_x, W_x = x.shape[2], x.shape[3]
        H_cond, W_cond = condition.shape[2], condition.shape[3]

        if H_cond != H_x or W_cond != W_x:
            condition_upsampled = F.interpolate(condition, size=(H_x, W_x), mode='bilinear', align_corners=False)
        else:
            condition_upsampled = condition


        gamma = self.conv_gamma(condition_upsampled) # 形状: [B, norm_channels, H_x, W_x]
        beta = self.conv_beta(condition_upsampled)   # 形状: [B, norm_channels, H_x, W_x]


        out = norm_x * (1 + gamma) + beta # 论文中是 (1 + gamma)，可以防止 gamma 初始过小

        return out
class CoordinateAttentionResnet(nn.Module):
    def __init__(self,ch_in,ch_out,reduction=16): #小，大
        super().__init__()

        self.smallzhilian_1 = nn.Sequential(

            conv2d(ch_in, ch_in // 2, kernel_size=1),  # 降维
            conv2d(ch_in // 2, ch_in // 2, 3, 1, 1),  # 保持分辨率，提取局部特征
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(ch_in // 2, ch_in, kernel_size=1),  # 升维
            conv2d(ch_in, ch_in, 3, 1, 1),  # 保持分辨率，进一步提取特征
            nn.AdaptiveAvgPool2d(1),  # 全局池化到1x1，此时空间信息已充分聚合到通道中
            nn.Sigmoid(),
            )
        self.smallzhilian_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4)),
            conv2d(ch_in, ch_in, 3, 1, 1),  # 通道扩展
            Swish(),
            conv2d(ch_in, ch_in, 3, 1, 1),  # 通道恢复
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1)),
        )
        self.yazha=nn.AdaptiveAvgPool2d((1))
        self.pool_h=nn.AdaptiveAvgPool2d((1,None))
        self.pool_w=nn.AdaptiveAvgPool2d((None,1))
        mid_channels=max(ch_in // reduction, 8)
        self.conv1=nn.Conv2d(ch_in,mid_channels,kernel_size=1,stride=1,padding=0)
        self.act=nn.LeakyReLU(0.2,inplace=True)
        self.conv1_feat=conv2d(ch_in,ch_out,kernel_size=1)
        self.conv_h = nn.Conv2d(mid_channels, ch_in, kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(mid_channels, ch_in, kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
    def forward(self,feat_small,feat_big):
        B, C, H, W = feat_small.shape
        x_h = self.pool_h(feat_small)  # [B, C, 1, W]
        x_w = self.pool_w(feat_small)  # [B, C, H, 1]
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, 1, H]

        x_cat = torch.cat([x_h, x_w], dim=3)  # [B, C, 1, 18]


        x_cat = self.act(self.bn1(self.conv1(x_cat)))  # [B, mid_channels, 1, 18]


        x_h, x_w = torch.split(x_cat, [H, W], dim=3)  # x_h: (B, mid, 1, 2), x_w: (B, mid, 1,16)

        x_h = x_h.permute(0, 1, 3, 2)  # [B, mid, 2, 1]
        x_w = x_w.permute(0, 1, 2, 3)  # [B, mid, 1,16]


        attn_h = self.conv_h(x_h).sigmoid()  # [B, ch_out, 2, 1]
        attn_w = self.conv_w(x_w).sigmoid()  # [B, ch_out, 1,16]


        feat_small_R = self.smallzhilian_1(feat_small)
        feat_small_R2=self.smallzhilian_2(feat_small)

        mid_out = feat_small_R* attn_w * attn_h*feat_small_R2

        mid_out=self.yazha(mid_out)
        fin_out=self.conv1_feat(mid_out)*feat_big

        return fin_out+feat_big

