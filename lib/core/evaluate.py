# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def rmse_metric(preds, target):
    """
    Compute average RMSE and per-joint RMSE between predicted and ground-truth keypoints.

    Args:
        preds: numpy array of shape (batch_size, num_joints, 2)
        target: numpy array of shape (batch_size, num_joints, 2)

    Returns:
        avg_rmse: float — overall RMSE across all joints and samples
        rmse_per_joint: numpy array of shape (num_joints,) — RMSE per joint
    """
    assert preds.shape == target.shape, "Shape mismatch between predictions and targets"
    
    # Valid points: where both x and y are > 1
    valid = (target[..., 0] > 1) & (target[..., 1] > 1)
    
    # Squared differences
    squared_diff = (preds - target) ** 2  # (batch, joints, 2)
    sum_squared = np.sum(squared_diff, axis=2)  # (batch, joints)

    # Zero out invalid joints
    sum_squared[~valid] = 0

    # Count valid joints
    valid_counts = np.sum(valid, axis=0)  # (num_joints,)

    # Per-joint RMSE (avoid division by zero)
    rmse_per_joint = np.zeros(preds.shape[1])
    for j in range(preds.shape[1]):
        if valid_counts[j] > 0:
            rmse_per_joint[j] = np.sqrt(np.sum(sum_squared[:, j]) / valid_counts[j])
        else:
            rmse_per_joint[j] = np.nan  # or 0 if you prefer

    # Overall RMSE
    total_valid = np.sum(valid)
    if total_valid > 0:
        avg_rmse = np.sqrt(np.sum(sum_squared) / total_valid)
    else:
        avg_rmse = np.nan

    return avg_rmse, rmse_per_joint



def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


