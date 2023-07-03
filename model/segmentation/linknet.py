import torch
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS
from torch import nn
from torch.nn import functional as F
from torchvision import models


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


@MODELS.register_module()
class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, loss=None, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        filters = [64, 128, 256]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # Decoder
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        # load pretrained weights
        if pretrained is not None:
            pretrained_weights = torch.load(pretrained)['model']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_weights.items():
                k_ = k.replace('module.', '')
                if k_ == 'finalconv3.weight' or k_ == 'finalconv3.bias':
                    continue
                new_state_dict[k_] = v
            self.load_state_dict(new_state_dict, strict=False)
            print("load success")

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print("PARAM STATUS: total number of parameters %.3fM in segmentation network" % (n_parameters / 1000 ** 2))

    # noinspection PyCallingNonCallable
    def forward(self, left_inp, img_meta, state, outputs):
        # Encoder
        x = self.firstconv(left_inp)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        x_out = F.softmax(f5, dim=1)[:, 1:]  # skip background class
        outputs['pred_semantic_seg'] = x_out
        outputs['pred_semantic_seg_raw'] = f5
        return outputs

    def losses(self, loss, outputs, state, meta, **kwargs):
        gt_segm = state['gt_semantic_seg']
        pred_segm_raw = outputs['pred_semantic_seg_raw']
        invalid_mask = state.get('invalid_mask', None)
        self.loss(gt_segm, pred_segm_raw, loss, meta, invalid_mask)

    def freeze(self):
        self.eval()
        self.loss.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.freezed = True
