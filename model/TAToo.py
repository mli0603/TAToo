import copy
import os.path as osp
from abc import ABCMeta
from collections import OrderedDict

import lietorch
import numpy as np
import torch
import torch.distributed as dist
from lietorch import SE3
from mmcv.runner import BaseModule, auto_fp16
from mmcv.utils import mkdir_or_exist
from mmseg.models.builder import MODELS

from utils import compute_valid_mask, thres_metric, pose_acc, pose_acc_rel, segmentation_metrics, merge_seg
from .builder import ESTIMATORS
from .motion.others import compute_flow


def calc_metric(output, state, meta):
    img_h, img_w = meta["img_shape"]  # to remove padded region for eval

    results = OrderedDict()

    if 'gt_disp' in state and 'pred_disp' in output:
        gt_disp = state['gt_disp'][..., :img_h, :img_w]
        pred_disp = output['pred_disp'][..., :img_h, :img_w]
        # mask excludes invalid disp
        mask_disp = compute_valid_mask(gt_disp=gt_disp, meta=meta)

        # disparity
        if mask_disp.any():  # only compute metrics if there are valid pixels
            # compute metrics
            results["epe"] = torch.mean(
                torch.abs(pred_disp[mask_disp] - gt_disp[mask_disp]))

            results["th3"] = thres_metric(pred_disp, gt_disp, mask_disp, 3.0)

            # depth error
            fx = state['intrinsics'][..., 0, None, None]
            results["depth"] = torch.mean(
                torch.abs(meta['baseline'] * fx / pred_disp[mask_disp] - meta['baseline'] * fx / gt_disp[mask_disp]))
        else:
            results["epe"] = np.NAN
            results["depth"] = np.NAN
            results["th3"] = np.NAN

    # segmentation
    if 'gt_semantic_seg' in state and 'pred_semantic_seg' in output:
        pred_semantic_seg = output['pred_semantic_seg'][..., :img_h, :img_w]
        gt_semantic_seg = state['gt_semantic_seg'][..., :img_h, :img_w]

        iou, acc, dice = segmentation_metrics(
            pred_semantic_seg, gt_semantic_seg)
        results['iou'] = iou.mean()
        results['acc'] = acc.mean()
        results['dice'] = dice.mean()

        # patient anatomy
        results['iou_p'] = iou[0]
        results['acc_p'] = acc[0]
        results['dice_p'] = dice[0]

        # surgical drill
        results['iou_d'] = iou[1]
        results['acc_d'] = acc[1]
        results['dice_d'] = dice[1]

    # pose
    if 'gt_pose_cam_patient' in state and 'pred_pose_cam_patient' in output:
        gt_pose_cam_patient = state['gt_pose_cam_patient']
        gt_pose_cam_drill = state['gt_pose_cam_drill']
        pred_pose_cam_patient = output['pred_pose_cam_patient']
        pred_pose_cam_drill = output['pred_pose_cam_drill']

        # result
        results.update(pose_acc(gt_pose_cam_patient,
                       pred_pose_cam_patient, obj='p'))
        results.update(pose_acc(gt_pose_cam_drill,
                       pred_pose_cam_drill, obj='d'))
        results.update(pose_acc_rel(gt_pose_cam_drill, gt_pose_cam_patient, pred_pose_cam_drill, pred_pose_cam_patient))

        # evaluate pose induced scene flow
        if 'pred_flow' in output:
            # flow
            Gs=lietorch.cat([gt_pose_cam_patient, gt_pose_cam_drill], dim = 0)
            M=Gs.shape[1]
            ii=torch.arange(M - 1)
            jj=ii + 1
            fx=state['intrinsics'][..., 0, None, None]
            disp=output['pred_disp'] / meta['baseline'] / fx
            seg=merge_seg(output['pred_semantic_seg'])
            intrinsics=state['intrinsics']
            # NOTE: this needs to be of shape BxNxHxWx3, and last dimension is not scaled up
            gt_flow=compute_flow(
                [Gs], disp, seg, intrinsics, ii, jj, meta)[-1]

            pred_flow=output['pred_flow']
            mask_flow=compute_valid_mask(gt_flow = gt_flow, meta = meta)

            scale=torch.ones([pred_flow.shape[0], 1, 1, 1, 3],
                               device = pred_flow.device)  # BxNxHxWx3
            scale[..., -1]=meta['baseline'] * \
                state['intrinsics'][:, ii, 0, None, None]  # convert to pixel

            err=scale * (pred_flow - gt_flow).abs()
            gt_semantic_seg=state['gt_semantic_seg']
            patient_mask=gt_semantic_seg[:, 0, [0]].bool()
            drill_mask=gt_semantic_seg[:, 0, [1]].bool()
            results['scene_flow_epe_x']=err[mask_flow][:, 0].mean()
            results['scene_flow_epe_x_p']=err[mask_flow &
                                                patient_mask][:, 0].mean()
            results['scene_flow_epe_x_d'] = err[mask_flow &
                                                drill_mask][:, 0].mean()
            results['scene_flow_epe_y'] = err[mask_flow][:, 1].mean()
            results['scene_flow_epe_y_p'] = err[mask_flow &
                                                patient_mask][:, 1].mean()
            results['scene_flow_epe_y_d'] = err[mask_flow &
                                                drill_mask][:, 1].mean()
            results['scene_flow_epe_z'] = err[mask_flow][:, 2].mean()
            results['scene_flow_epe_z_p'] = err[mask_flow &
                                                patient_mask][:, 2].mean()
            results['scene_flow_epe_z_d'] = err[mask_flow &
                                                drill_mask][:, 2].mean()

        # evaluate predicted updates, at 1/8 resolution
        if 'pred_target' in output:
            # flow
            Gs = lietorch.cat([gt_pose_cam_patient, gt_pose_cam_drill], dim=0)
            M = Gs.shape[1]
            ii = torch.arange(M - 1)
            jj = ii + 1
            fx = state['intrinsics'][..., 0, None, None]
            disp = output['pred_disp'] / meta['baseline'] / fx
            seg = merge_seg(output['pred_semantic_seg'])
            intrinsics = state['intrinsics']

            disp_lr = disp[..., 3: : 8, 3: : 8]
            intrinsics_lr = intrinsics / 8.0
            seg_lr = seg[..., 3: : 8, 3: : 8]
            meta_lr = copy.deepcopy(meta)
            meta_lr['disp_range'] = (
                meta['disp_range'][0] / 8.0, meta['disp_range'][1] / 8.0)

            # NOTE: this needs to be of shape BxNxHxWx3, and last dimension is not scaled up
            gt_target=compute_flow(
                [Gs], disp_lr, seg_lr, intrinsics_lr, ii, jj, meta_lr, return_target = True)[-1]
            pred_target=output['pred_target'][-1]

            mask_target=gt_target[1] & pred_target[1]
            mask_target[..., img_h // 8:, :, :]=False
            mask_target[..., img_w // 8:, :]=False
            mask_target=mask_target.squeeze(-1)
            gt_target=gt_target[0]
            pred_target=pred_target[0]

            err=(pred_target - gt_target).abs()
            gt_semantic_seg=state['gt_semantic_seg'][..., 3::8, 3::8]
            patient_mask=gt_semantic_seg[:, 0, [0]].bool()
            drill_mask=gt_semantic_seg[:, 0, [1]].bool()

            results['optical_flow_epe_x']=err[mask_target][:, 0].mean()
            results['optical_flow_epe_x_p']=err[mask_target &
                                                  patient_mask][:, 0].mean()
            results['optical_flow_epe_x_d'] = err[mask_target &
                                                  drill_mask][:, 0].mean()
            results['optical_flow_epe_y'] = err[mask_target][:, 1].mean()
            results['optical_flow_epe_y_p'] = err[mask_target &
                                                  patient_mask][:, 1].mean()
            results['optical_flow_epe_y_d'] = err[mask_target &
                                                  drill_mask][:, 1].mean()

    return results


def prepare_state(kwargs):
    state = OrderedDict()

    # compute relative poses
    gt_pose_cam = kwargs.get('gt_pose_cam', None)
    gt_pose_drill = kwargs.get('gt_pose_drill', None)
    gt_pose_patient = kwargs.get('gt_pose_patient', None)

    if gt_pose_cam is not None:
        gt_pose_cam = SE3.InitFromVec(gt_pose_cam)  # T_w_c
        gt_pose_patient = SE3.InitFromVec(gt_pose_patient)  # T_w_p
        gt_pose_drill = SE3.InitFromVec(gt_pose_drill)  # T_w_t

        state['gt_pose_cam_patient'] = gt_pose_cam.inv(
        ) * gt_pose_patient  # T_c_p, NOTE: BxM
        state['gt_pose_cam_drill']=gt_pose_cam.inv(
        ) * gt_pose_drill  # T_c_d, NOTE: BxM

    # intrinsics
    state['intrinsics']=kwargs.get('intrinsics', None)  # NOTE: BxMx4

    # disp and seg
    if 'gt_disp' in kwargs:
        # NOTE: this needs to be of shape BxMxHxW
        state['gt_disp']=kwargs["gt_disp"]
    if 'gt_semantic_seg' in kwargs:
        # NOTE: this needs to be of shape BxMxobjxHxW
        state['gt_semantic_seg']=kwargs['gt_semantic_seg']
    if 'invalid_mask' in kwargs:
        state['invalid_mask']=kwargs['invalid_mask']

    return state


@ ESTIMATORS.register_module()
class TAToo(BaseModule, metaclass = ABCMeta):
    """Consistent online depth network"""

    def __init__(self, stereo = None, motion = None, segmentation = None, renderer = None, train_cfg = None, test_cfg = None,
                 init_cfg = None, **kwargs):
        super(TAToo, self).__init__(**kwargs)
        self.fp16_enabled=False

        self.train_cfg=train_cfg
        self.test_cfg=test_cfg

        if stereo is not None:
            self.stereo=MODELS.build(stereo)
        else:
            self.stereo=None
        if motion is not None:
            self.motion=MODELS.build(motion)
        else:
            self.motion=None
        if segmentation is not None:
            self.segmentation=MODELS.build(segmentation)
        else:
            self.segmentation=None
        if renderer is not None:
            self.renderer=MODELS.build(renderer)
        else:
            self.renderer=None

    def freeze_motion(self):
        if (self.train_cfg is not None) and (self.train_cfg.get("freeze_motion", False)):
            return True
        else:
            return False

    def freeze_stereo(self):
        if (self.train_cfg is not None) and (self.train_cfg.get("freeze_stereo", False)):
            return True
        else:
            return False

    def freeze_segmentation(self):
        if (self.train_cfg is not None) and (self.train_cfg.get("freeze_segmentation", False)):
            return True
        else:
            return False

    def freeze_renderer(self):
        if (self.train_cfg is not None) and (self.train_cfg.get("freeze_renderer", False)):
            return True
        else:
            return False

    def estimate_pose(self, left, right, img_meta, state):
        """network

        Args:
            left (Tensor)
            right (Tensor)
            img_meta (Tensor): dataset metas
            state (dict): states storing past information

        Returns:
            dict: outputs
        """
        outputs=dict()
        B, M, _, H, W=left.shape
        device=left.device
        if self.stereo is not None:
            left_inp=left.view(B * M, -1, H, W)
            right_inp=right.view(B * M, -1, H, W)
            if self.freeze_stereo() or not self.training:
                with torch.no_grad():
                    outputs=self.stereo(
                        left_inp, right_inp, img_meta, state, outputs)
            else:
                outputs=self.stereo(
                    left_inp, right_inp, img_meta, state, outputs)
            # reshape
            outputs['pred_disp'] = outputs['pred_disp'].view(B, M, H, W)
            Hh, Ww = outputs['disp_confidence'].shape[-2:]
            outputs['disp_confidence'] = outputs['disp_confidence'].view(
                B, M, Hh, Ww)
            # find in range disp
            outputs['pred_disp_mask'] = compute_valid_mask(
                img_meta, gt_disp=outputs["pred_disp"])

        # NOTE: segmentation outputs probability over the two classes, not three classes
        if self.segmentation is not None:
            left_inp = left.view(B * M, -1, H, W)
            if self.freeze_segmentation() or not self.training:
                with torch.no_grad():
                    outputs = self.segmentation(
                        left_inp, img_meta, state, outputs)
            else:
                outputs = self.segmentation(left_inp, img_meta, state, outputs)
            # reshape
            outputs['pred_semantic_seg'] = outputs['pred_semantic_seg'].view(
                B, M, 2, H, W)

            # to remove padded region due to reflection
            img_h, img_w = img_meta["img_shape"]
            seg_valid_mask = torch.zeros_like(
                outputs['pred_semantic_seg'], device=device)
            seg_valid_mask[..., :img_h, :img_w] = 1.0
            invalid_mask = state.get('invalid_mask', None)
            if invalid_mask is not None:
                seg_valid_mask = (
                    1 - invalid_mask).unsqueeze(2) * seg_valid_mask
            outputs['pred_semantic_seg'] = outputs['pred_semantic_seg'] * \
                seg_valid_mask

        # NOTE: flow is of unit [px,px,normalized px], note the last dimension has never been scaled back
        if self.motion is not None:
            pred_semantic_seg = outputs['pred_semantic_seg']
            # convert disp to normalized space for numerical stability
            pred_disp = outputs['pred_disp']
            if self.freeze_motion() or not self.training:
                with torch.no_grad():
                    outputs = self.motion(left, pred_disp, pred_semantic_seg, outputs['disp_confidence'], img_meta,
                                          state, outputs)
            else:
                outputs = self.motion(left, pred_disp, pred_semantic_seg, outputs['disp_confidence'], img_meta, state,
                                      outputs)

        if self.renderer is not None:
            fx = state['intrinsics'][..., 0][..., None, None]
            pred_semantic_seg = outputs['pred_semantic_seg']
            # convert disp to normalized space for numerical stability
            pred_disp = outputs['pred_disp'] / img_meta['baseline'] / fx
            # given pose and flow, sample the target image into current frame
            if self.freeze_renderer() or not self.training:
                with torch.no_grad():
                    outputs = self.renderer(
                        left, pred_disp, pred_semantic_seg, img_meta, state, outputs)
            else:
                outputs = self.renderer(
                    left, pred_disp, pred_semantic_seg, img_meta, state, outputs)

        return outputs

    @auto_fp16(apply_to=("img", "r_img"))
    def forward(self, left, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]).
        """
        if return_loss:
            return self.forward_train(left, img_metas, **kwargs)
        else:
            return self.forward_test(left, img_metas, **kwargs)

    def forward_train(self, left, img_metas, right, **kwargs):
        train_state = prepare_state(kwargs)

        losses = dict()

        # forward pass
        output = self.estimate_pose(left, right, img_metas[0], train_state)
        if 'reg_data' in kwargs.keys():  # inference on reg data
            left_inp = kwargs['reg_data']['left']
            right_inp = kwargs['reg_data']['right']
            gt_disp = kwargs['reg_data']["gt_disp"]
            if len(left_inp.shape) == 5:
                left_inp = left_inp.squeeze(1)
                right_inp = right_inp.squeeze(1)
            reg_output = dict()
            reg_state = dict()
            reg_state['gt_disp'] = gt_disp
            reg_state['reg_loss_weight'] = kwargs['reg_data']['reg_loss_weight']
            if self.freeze_stereo() or not self.training:
                with torch.no_grad():
                    reg_output = self.stereo(
                        left_inp, right_inp, img_metas[0], reg_state, reg_output)
            else:
                reg_output = self.stereo(
                    left_inp, right_inp, img_metas[0], reg_state, reg_output)
            train_state['reg_state'] = reg_state
            output['reg_output'] = reg_output

        loss = self.losses(output, train_state, img_metas[0])
        losses.update(loss)

        return losses

    def losses(self, output, state, meta):
        """compute losses

        Args:
            output (List)
            state (dict): memory states of past information
            meta (List): dataset meta

        Returns:
            dict: losses
        """
        loss = dict()

        if self.stereo is not None and not self.freeze_stereo() and self.stereo.loss is not None:
            B, M, H, W = state['gt_disp'].shape
            state['gt_disp'] = state['gt_disp'].view(B * M, 1, H, W)
            self.stereo.losses(loss, output, state, meta)
            state['gt_disp'] = state['gt_disp'].view(B, M, H, W)

            if 'reg_state' in state.keys():
                reg_loss = dict()
                self.stereo.losses(
                    reg_loss, output['reg_output'], state['reg_state'], meta)
                for key in reg_loss.keys():
                    loss[key + '_reg'] = reg_loss[key] * \
                        state['reg_state']['reg_loss_weight']

        if self.segmentation is not None and not self.freeze_segmentation() and self.segmentation.loss is not None:
            self.segmentation.losses(loss, output, state, meta)

        if self.motion is not None and not self.freeze_motion() and self.motion.loss is not None:
            self.motion.losses(loss, output, state, meta)

        if self.renderer is not None and not self.freeze_renderer() and self.renderer.loss is not None:
            self.renderer.losses(loss, output, state, meta)

        return loss

    def forward_test(self, left, img_metas, right=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): The outer list is not used.
            img_metas (List[List[dict]]): The outer list is not used.
                The inner list indicates images in a batch.
        """

        with torch.no_grad():
            pred = self.inference(left, right, img_metas, **kwargs)

        # convert to list
        pred = [pred]
        return pred

    def inference(self, left, right, img_metas, evaluate=True, **kwargs):
        """inference

        Args:
            left (Tensor): left image
            right (Tensor): right image
            img_meta (List): dataset meta
            reciprocal (bool, optional): wheter prediction is depth, if True, use "calib" key in meta to convert to disparity. Defaults to False.
            evaluate (bool, optional): if True, evalue against GT, if False, output disparity for visualization. Defaults to True.

        Returns:
            Tensor: The output disp prediction (evaluate=False) or metrics (evaluate=True)
        """
        inference_state = prepare_state(kwargs)

        # forward pass
        output = self.estimate_pose(left, right, img_metas[0], inference_state)

        # perform evaluation if needed
        if evaluate:
            metrics = calc_metric(output, inference_state, img_metas[0])
            return metrics
        else:  # otherwise, return predictions
            return output

    @staticmethod
    def show_result(filename, result, show=False, out_file=None, running_stats=None, **kwargs):
        """show result either to terminal or save output

        Args:
            filename (str)
            result (Tensor): disparity or metrics
            show (bool, optional): if show, output disparity. Defaults to False.
            out_file (str, optional): output filename. Defaults to None.
            running_stats (optional): running stats to accumulate results. Defaults to None.
        """
        if not show and running_stats:
            result = result[0]
            if running_stats.header is None:
                running_stats.header = ["filename"] + \
                    [k for k in result.keys()]
            running_stats.push(
                filename, [result[k].cpu().mean().item() for k in result.keys()])
        else:
            output = result[0]
            mkdir_or_exist(osp.dirname(out_file))
            if 'pred_semantic_seg' in output:
                np.savez_compressed(
                    out_file + '_segmentation', data=output['pred_semantic_seg'].cpu().numpy() > 0.75)
            if 'pred_disp' in output:
                np.savez_compressed(
                    out_file + '_disp', data=output['pred_disp'].cpu().numpy().astype(np.half))
            if 'pred_pose_cam_patient' in output:
                np.savez_compressed(out_file + '_pose_cam_patient',
                                    data=output['pred_pose_cam_patient'].cpu().data.numpy().astype(np.half))
            if 'pred_pose_cam_drill' in output:
                np.savez_compressed(out_file + '_pose_cam_drill',
                                    data=output['pred_pose_cam_drill'].cpu().data.numpy().astype(np.half))
            if 'pred_flow' in output:
                np.savez_compressed(
                    out_file + '_flow', data=output['pred_flow'].cpu().data.numpy().astype(np.half))

    def train(self, mode=True):
        """overloading torch's train function to freeze different modules when necessary

        Args:
            mode (bool, optional): True to train, False to eval. Defaults to True.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if mode is False:
            return

        if self.freeze_stereo() and self.stereo is not None:
            self.stereo.freeze()

        if self.freeze_segmentation() and self.segmentation is not None:
            self.segmentation.freeze()

        if self.freeze_motion() and self.motion is not None:
            self.motion.freeze()

        if mode:
            n_parameters = sum(p.numel()
                               for n, p in self.named_parameters() if p.requires_grad)
            print("PARAM STATUS: total number of training parameters %.3fM" %
                  (n_parameters / 1000 ** 2))

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        train_epe_attrs = [attr for attr in dir(self) if "train_epe" in attr]
        for attr in train_epe_attrs:
            log_vars.update({attr: getattr(self, attr)})

        outputs = dict(loss=loss, log_vars=log_vars,
                       num_samples=len(data_batch["left"].data))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for k, v in loss_value.items():
                    log_vars[loss_name + "_" + k] = v
            else:
                raise TypeError(
                    f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if
                   _key.startswith("loss") or (_key.startswith("decode") and "loss" in _key))

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
