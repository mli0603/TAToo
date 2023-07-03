import math
import random

import albumentations.augmentations.functional as F
import cv2
import numpy as np
import torch
import torch.nn.functional as TF
from albumentations import ColorJitter, ToGray, Compose
from albumentations.core.transforms_interface import BasicTransform
from mmseg.datasets import PIPELINES

"""
Base
"""


class StereoTransform(BasicTransform):
    """
    Transform applied to image only.
    """

    @property
    def targets(self):
        return {
            "left": self.apply,
            "right": self.apply
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["left"].shape[1], "rows": kwargs["right"].shape[0]})
        return params


class RightOnlyTransform(BasicTransform):
    """
    Transform applied to right image only.
    """

    @property
    def targets(self):
        return {
            "right": self.apply
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["right"].shape[1], "rows": kwargs["right"].shape[0]})
        return params


class StereoTransformAsym(BasicTransform):
    """
    Transform applied not equally to left and right images.
    """

    def __init__(self, always_apply=False, p=0.5, p_asym=0.2):
        super(StereoTransformAsym, self).__init__(always_apply, p)
        self.p_asym = p_asym

    @property
    def targets(self):
        return {
            "left": self.apply_l,
            "right": self.apply_r
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["left"].shape[1], "rows": kwargs["right"].shape[0]})
        return params

    @property
    def targets_as_params(self):
        return ["left", "right"]

    def asym(self):
        return random.random() < self.p_asym
        # return False


"""
Stereo Image only transform
"""


class BGRToRGB(StereoTransform):
    def __init__(self, always_apply=True, p=1.0):
        super(BGRToRGB, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return image[..., ::-1].copy()


class Normalize(StereoTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        left, right

    Image types:
        uint8, float32
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True,
                 p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")


class ToGrayStereo(StereoTransform, ToGray):
    def __init__(self, always_apply=False, p=0.5):
        StereoTransform.__init__(self, always_apply, p)
        ToGray.__init__(self, always_apply, p)


"""
Stereo Image Only Asym Transform
"""


class ColorJitterStereo(StereoTransformAsym, ColorJitter):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5, p_asym=0.2):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        ColorJitter.__init__(self, brightness, contrast, saturation, hue, always_apply, p)

    def apply_l(self, img, l_transform, **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")

        for transform in l_transform:
            img = transform(img)
        return img

    def apply_r(self, img, r_transform, **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")

        for transform in r_transform:
            img = transform(img)
        return img

    def get_params_dependent_on_targets(self, params):
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        transforms = [
            lambda x: F.adjust_brightness_torchvision(x, brightness),
            lambda x: F.adjust_contrast_torchvision(x, contrast),
            lambda x: F.adjust_saturation_torchvision(x, saturation),
            lambda x: F.adjust_hue_torchvision(x, hue),
        ]
        random.shuffle(transforms)

        if self.asym():
            brightness = random.uniform(self.brightness[0], self.brightness[1])
            contrast = random.uniform(self.contrast[0], self.contrast[1])
            saturation = random.uniform(self.saturation[0], self.saturation[1])
            hue = random.uniform(self.hue[0], self.hue[1])

            r_transforms = [
                lambda x: F.adjust_brightness_torchvision(x, brightness),
                lambda x: F.adjust_contrast_torchvision(x, contrast),
                lambda x: F.adjust_saturation_torchvision(x, saturation),
                lambda x: F.adjust_hue_torchvision(x, hue),
            ]
            random.shuffle(r_transforms)
        else:
            r_transforms = transforms

        return {
            "l_transform": transforms,
            "r_transform": r_transforms
        }


"""
Right Image Only
"""


class RandomShiftRotate(RightOnlyTransform):
    """Randomly apply vertical translate and rotate the input.
    Args:
        max_shift (float): maximum shift in pixels along vertical direction. Default: 1.5.
        max_rotation (float): maximum rotation in degree. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(self, max_shift=1.5, max_rotation=0.2, always_apply=False, p=1.0):
        super(RandomShiftRotate, self).__init__(always_apply, p)
        self.max_shift = max_shift
        self.max_rotation = max_rotation

    def apply(self, img, **params):
        h, w, _ = img.shape
        shift = random.random() * self.max_shift * 2 - self.max_shift
        rotation = random.random() * self.max_rotation * 2 - self.max_rotation

        matrix = np.float32([[np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation)), 0],
                             [np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation)), shift]])

        return cv2.warpAffine(img, matrix, (w, h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


@PIPELINES.register_module(force=True)
class StereoAugmentor:
    """ perform augmentation on stereo video. Adapted from Droid-slam. """

    def __init__(self, crop_size, bgr_to_rgb=True):
        self.crop_size = crop_size
        aug_list = [
            ColorJitterStereo(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4 / 3.14, p=0.5),
            ToGrayStereo(p=0.1),
            # RandomShiftRotate(p=0.5),
            Normalize(p=1.0),
        ]
        if bgr_to_rgb:
            aug_list.insert(0, BGRToRGB(p=1.0))

        self.augcolor = Compose(aug_list)

        self.max_scale = 0.25

    def spatial_transform(self, results):
        left = results['left']
        right = results['right']
        intrinsics = results.get('intrinsics', None)
        gt_disp = results.get('gt_disp', None)
        gt_semantic_seg = results.get('gt_semantic_seg', None)
        invalid_mask = results.get('invalid_mask', None)

        """ cropping and resizing """
        ht, wd = left.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 2 ** np.random.uniform(min_scale, max_scale)

        left = TF.interpolate(left, scale_factor=scale, mode='bilinear', align_corners=False,
                              recompute_scale_factor=True)
        right = TF.interpolate(right, scale_factor=scale, mode='bilinear', align_corners=False,
                               recompute_scale_factor=True)
        sx = left.shape[-1] / float(wd)
        sy = left.shape[-2] / float(ht)
        if intrinsics is not None:
            intrinsics = intrinsics * torch.as_tensor([sx, sy, sx, sy])

        y0 = (left.shape[-2] - self.crop_size[0]) // 2
        x0 = (left.shape[-1] - self.crop_size[1]) // 2
        if intrinsics is not None:
            intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        left = left[..., y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        right = right[..., y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        if gt_semantic_seg is not None:
            gt_semantic_seg = TF.interpolate(gt_semantic_seg, scale_factor=scale, mode='nearest',
                                             recompute_scale_factor=True)
            gt_semantic_seg = gt_semantic_seg[..., y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            results['gt_semantic_seg'] = gt_semantic_seg
        if gt_disp is not None:
            gt_disp = gt_disp.unsqueeze(dim=1)
            gt_disp = TF.interpolate(gt_disp, scale_factor=scale, mode='nearest', recompute_scale_factor=True)
            gt_disp = gt_disp.squeeze(dim=1)
            gt_disp = sx * gt_disp
            gt_disp = gt_disp[..., y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            results['gt_disp'] = gt_disp
        if invalid_mask is not None:
            invalid_mask = invalid_mask.unsqueeze(dim=1)
            invalid_mask = TF.interpolate(invalid_mask, scale_factor=scale, mode='nearest', recompute_scale_factor=True)
            invalid_mask = invalid_mask.squeeze(dim=1)
            invalid_mask = invalid_mask[..., y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            results['invalid_mask'] = invalid_mask

        results['left'] = left
        results['right'] = right
        if intrinsics is not None:
            results['intrinsics'] = intrinsics
        return results

    def color_transform(self, results):
        """ color jittering """
        num, ht, wd, ch = results['left'].shape

        left = np.zeros_like(results['left']).astype(np.float32)
        right = np.zeros_like(results['right']).astype(np.float32)

        for i in range(num):
            temp_result = dict(left=results['left'][i], right=results['right'][i])
            temp_result = self.augcolor(**temp_result)
            left[i] = temp_result['left']
            right[i] = temp_result['right']

        results['left'] = left.transpose(0, 3, 1, 2)  # MxCxHxW
        results['right'] = right.transpose(0, 3, 1, 2)  # MxCxHxW
        return results

    def totensor(self, results):
        for k, v in results.items():
            results[k] = torch.from_numpy(v).contiguous().float()
        return results

    def __call__(self, results):
        results = self.color_transform(results)
        results = self.totensor(results)
        return self.spatial_transform(results)


@PIPELINES.register_module(force=True)
class StereoNormalizor(StereoAugmentor):
    """ perform augmentation on stereo video. Adapted from Droid-slam. """

    def __init__(self, divisor=64, bgr_to_rgb=True):
        aug_list = [
            Normalize(p=1.0),
        ]
        if bgr_to_rgb:
            aug_list.insert(0, BGRToRGB(p=1.0))

        self.augcolor = Compose(aug_list)
        self.divisor = divisor

        return

    def spatial_transform(self, results):
        H, W = results['left'].shape[2:]
        h_pad = math.ceil(H / self.divisor) * self.divisor - H
        w_pad = math.ceil(W / self.divisor) * self.divisor - W

        results['left'] = TF.pad(results['left'], (0, w_pad, 0, h_pad), 'constant', value=2.45)
        results['right'] = TF.pad(results['right'], (0, w_pad, 0, h_pad), 'constant', value=2.45)
        if 'invalid_mask' in results:
            results['invalid_mask'] = TF.pad(results['invalid_mask'], (0, w_pad, 0, h_pad), 'constant', value=1.0)
        if 'gt_disp' in results:
            results['gt_disp'] = TF.pad(results['gt_disp'], (0, w_pad, 0, h_pad), 'constant', value=0.0)
        if 'gt_semantic_seg' in results is not None:
            results['gt_semantic_seg'] = TF.pad(results['gt_semantic_seg'], (0, w_pad, 0, h_pad), 'constant', value=0.0)
        if 'gt_flow' in results is not None:
            results['gt_flow'] = TF.pad(results['gt_flow'], (0, w_pad, 0, h_pad), 'constant', value=500.0)

        return results

    def __call__(self, results):
        results = self.color_transform(results)
        results = self.totensor(results)
        return self.spatial_transform(results)
