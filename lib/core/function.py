
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from core.evaluate import accuracy
from core.evaluate import rmse_metric
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis_skeleton import cv2_visualize_keypoints


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    rmse_student = AverageMeter()
    rmse_teacher = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # compute output
        outputs, _ = model(input)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #       'Speed {speed:.1f} samples/s\t' \
            #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            #       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #           epoch, i, len(train_loader), batch_time=batch_time,
            #           speed=input.size(0)/batch_time.val,
            #           data_time=data_time, loss=losses, acc=acc)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, animalpose=False, vis=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    rmse_student = AverageMeter()
    rmse_teacher = AverageMeter()

    all_gts_list = []  # <-- ADD THIS

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    # all_preds = np.zeros(
    #     (num_samples, config.MODEL.NUM_JOINTS, 3),
    #     dtype=np.float32
    # )
    all_preds = []
    all_boxes = np.zeros((num_samples, 7)) if animalpose else np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0


    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs, _ = model(input)

            # all_gts_list.append(target.cpu().numpy())  # <-- ADD THIS INSIDE THE LOOP
            # all_gts_list.append(target[:, :, :, 0].cpu().numpy())  # just an example if heatmaps are (N, joints, H, W)
            # target shape: (batch_size, num_joints, height, width)
            target_np = target.cpu().numpy()
            batch_size, num_joints, h, w = target_np.shape

            # Initialize array for keypoint coordinates: (batch_size, num_joints, 2)
            gt_coords = np.zeros((batch_size, num_joints, 2), dtype=np.float32)

            for b in range(batch_size):
                for j in range(num_joints):
                    heatmap = target_np[b, j, :, :]
                    # Find argmax position in heatmap (flattened index)
                    idx = heatmap.argmax()
                    y, x = np.unravel_index(idx, (h, w))
                    gt_coords[b, j, :] = [x, y]  # Note order: (x, y)

            all_gts_list.append(gt_coords)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, _ = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            # all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # Concatenate preds and maxvals to shape (batch_size, num_joints, 3)
            batch_preds = np.concatenate([preds[:, :, 0:2], maxvals], axis=2)  # shape (batch_size, num_joints, 3)
            all_preds.append(batch_preds)



            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            # savepath = "/home/sammych/ScarceNet/output/output_animal_hrnet5_part/results"
            savepath = "output/output_animal_hrnet5_part/results"

            if animalpose:
                bbox_ids = meta['bbox_id'].numpy()
                all_boxes[idx:idx + num_images, 6] = bbox_ids
            if vis:
                cv2_visualize_keypoints(input, pred *4, savepath, idx)

            image_path.extend(meta['image'])
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        # AFTER THE LOOP
        # all_gts = np.concatenate(all_gts_list, axis=0)
        # preds_xy = all_preds[:, :, :2]
        # gts_xy = all_gts[:, :, :2]
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts_list, axis=0)
        preds_xy = all_preds[:, :, :2]


        # print("preds_xy shape:", preds_xy.shape)
        # print("gts_xy shape:", gts_xy.shape)
        # assert preds_xy.shape == gts_xy.shape, "Shape mismatch"

        # print("preds_xy contains NaNs:", np.isnan(preds_xy).any())
        # print("gts_xy contains NaNs:", np.isnan(gts_xy).any())

        # print("preds_xy min/max:", np.min(preds_xy), np.max(preds_xy))
        # print("gts_xy min/max:", np.min(gts_xy), np.max(gts_xy))

        # print("target shape inside loop:", target.shape)

        all_preds = np.concatenate(all_preds, axis=0)

        all_gts = np.concatenate(all_gts_list, axis=0)

        preds_xy = all_preds[:, :, :2]
        gts_xy = all_gts

        avg_rmse, rmse_per_joint = rmse_metric(preds_xy, gts_xy)
        logger.info('RMSE of keypoints (overall): {:.4f}'.format(avg_rmse))
        logger.info('RMSE per joint: {}'.format(rmse_per_joint))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_rmse', avg_rmse, global_steps)

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1
    return perf_indicator


# evaluate both student and teacher models
def validate_mt(config, val_loader, val_dataset, model, model_ema, criterion, output_dir,
             tb_log_dir, writer_dict=None, animalpose=False, vis=True):
    batch_time = AverageMeter()
    losses_sup = AverageMeter()
    losses_const = AverageMeter()
    acc = AverageMeter()
    acc_ema = AverageMeter()
    rmse_student = AverageMeter()
    rmse_teacher = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_ema.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_preds_ema = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 7)) if animalpose else np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs, _ = model(input)
            outputs_ema, _ = model_ema(input)
            if isinstance(outputs, list):
                output = outputs[-1]
                output_ema = outputs_ema[-1]
            else:
                output = outputs
                output_ema = outputs_ema
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, _ = model(input_flipped)
                outputs_flipped_ema, _ = model_ema(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                    output_flipped_ema = outputs_flipped_ema[-1]
                else:
                    output_flipped = outputs_flipped
                    output_flipped_ema = outputs_flipped_ema
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output_flipped_ema = flip_back(output_flipped_ema.cpu().numpy(),
                                               val_dataset.flip_pairs)
                output_flipped_ema = torch.from_numpy(output_flipped_ema.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                    output_flipped_ema[:, :, :, 1:] = \
                        output_flipped_ema.clone()[:, :, :, 0:-1]
                output = (output + output_flipped) * 0.5
                output_ema = (output_ema + output_flipped_ema) * 0.5
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            const_weight = torch.ones_like(target_weight).cuda()
            loss_sup = criterion(output, target, target_weight)
            loss_const =criterion(output, output_ema, const_weight)
            num_images = input.size(0)

            savepath = "output/output_animal_hrnet5_part/results_student_teacher"
            if vis:
                cv2_visualize_keypoints(input, pred *4, savepath, idx)
            # measure accuracy and record loss
            losses_sup.update(loss_sup.item(), num_images)
            losses_const.update(loss_const.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            _, avg_acc_ema, cnt_ema, pred_ema = accuracy(output_ema.cpu().numpy(),
                                                         target.cpu().numpy())
            
            acc.update(avg_acc, cnt)
            acc_ema.update(avg_acc_ema, cnt_ema)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            
            preds_ema, maxvals_ema = get_final_preds(
                config, output_ema.clone().cpu().numpy(), c, s)

            # Decode ground-truth keypoints
            target_coords, _ = get_final_preds(config, target.cpu().numpy(), c, s)

            # Compute RMSE
            rmse_batch = np.sqrt(np.mean((preds[:, :, 0:2] - target_coords[:, :, 0:2]) ** 2))
            rmse_student.update(rmse_batch, num_images)

            rmse_batch_ema = np.sqrt(np.mean((preds_ema[:, :, 0:2] - target_coords[:, :, 0:2]) ** 2))
            rmse_teacher.update(rmse_batch_ema, num_images)

            rmse_jointwise = np.sqrt(np.mean((preds[:, :, 0:2] - target_coords[:, :, 0:2]) ** 2, axis=0))  # shape: (num_joints, 2)


            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            preds_ema, maxvals_ema = get_final_preds(config, output_ema.clone().cpu().numpy(), c, s)
            all_preds_ema[idx:idx + num_images, :, 0:2] = preds_ema[:, :, 0:2]
            all_preds_ema[idx:idx + num_images, :, 2:3] = maxvals_ema

            if animalpose:
                bbox_ids = meta['bbox_id'].numpy()
                all_boxes[idx:idx + num_images, 6] = bbox_ids

            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss_sup {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Loss_const {loss_const.val:.4f} ({loss_const.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                      'Accuracy_ema {acc_ema.val:.3f} ({acc_ema.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses_sup, loss_const=losses_const, acc=acc, acc_ema=acc_ema)
                logger.info(msg)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        name_values_ema, perf_indicator_ema = val_dataset.evaluate(
            config, all_preds_ema, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if isinstance(name_values_ema, list):
            for name_value in name_values_ema:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_ema, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses_sup.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

            # # ---- Optional: Plot and Save Joint-wise RMSE ----
            # num_joints = rmse_jointwise.shape[0]
            # joint_indices = np.arange(num_joints)
            # rmse_jointwise_total = np.sqrt(np.mean((all_preds[:, :, 0:2] - all_preds_ema[:, :, 0:2]) ** 2, axis=0))
            # rmse_jointwise_per_joint = np.linalg.norm(rmse_jointwise_total, axis=1)

            # plt.figure(figsize=(12, 6))
            # plt.bar(joint_indices, rmse_jointwise_per_joint)
            # plt.xlabel('Joint Index')
            # plt.ylabel('RMSE')
            # plt.title('Joint-wise RMSE: Student vs Teacher')
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(f"{output_dir}/jointwise_rmse.png")
            # plt.close()

            # # ---- Optional: Log RMSE to TensorBoard ----
            # if writer_dict:
            #     for j in range(num_joints):
            #         writer.add_scalar(f'RMSE/joint_{j}', rmse_jointwise_per_joint[j], global_steps)

    return perf_indicator_ema


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
