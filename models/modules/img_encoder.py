import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from utils.base_util import pack, unpack

# https://github.com/pansanity666/TransHuman/blob/main/lib/networks/encoder.py#L50
class Resnet18Encoder(nn.Module):
    def __init__(self, module_cfg):
        super().__init__()
        self.cfg = module_cfg

        weights = ResNet18_Weights.DEFAULT
        pretrained_resnet18 = resnet18(weights=weights, progress=False)

        self.encoder = nn.Sequential(
            pretrained_resnet18.conv1,
            pretrained_resnet18.bn1,
            pretrained_resnet18.relu,
            pretrained_resnet18.maxpool,
            pretrained_resnet18.layer1,
            pretrained_resnet18.layer2,
            pretrained_resnet18.layer3,
            pretrained_resnet18.layer4)
        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        if module_cfg.rgb_embed:
            self.rgb_embed = nn.Conv2d(3, 128, 1)

    def train(self, mode=True):
        # freeze batchnorms
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.encoder.apply(set_bn_eval)

    def forward(self, imgs, masks, tb=None, **kwargs):
        if tb is not None:
            tb.summ_images('img_encoder/inputs', imgs.permute(0, 1, 4, 2, 3)[0])

        B, N, H, W, C = imgs.shape
        imgs = pack(imgs).permute(0, 3, 1, 2)
        imgs = (imgs - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        x = imgs

        x0 = self.encoder[2](self.encoder[1](self.encoder[0](x)))
        feats = [unpack(x0, [B, N])]

        x1 = self.encoder[3](x0)
        x1 = self.encoder[4](x1)
        feats.append(unpack(x1, [B, N]))

        x2 = self.encoder[5](x1)
        feats.append(unpack(x2, [B, N]))

        x3 = self.encoder[6](x2)
        feats.append(unpack(x3, [B, N]))

        x4 = self.encoder[7](x3)
        feats.append(unpack(x4, [B, N]))

        if self.cfg.rgb_embed:
            img_feat = self.rgb_embed(imgs)
            feats.append(unpack(img_feat, [B, N]))

        if tb is not None:
            tb.summ_feats('img_encoder/outputs', feats[-1][0])
        return feats
