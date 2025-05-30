
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import _init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.function import validate
from lib.utils.utils import create_logger
# import lib.models
from lib.models import pose_hrnet_part
import lib.dataset_animal

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--animalpose',
                        help='train on ap10k',
                        action='store_true')
    parser.add_argument('--vis', action='store_true')



    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.animalpose:
        valid_dataset = eval('lib.dataset_animal.' + cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        valid_dataset = eval('lib.dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    
    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir, animalpose=args.animalpose, vis=args.vis)


if __name__ == '__main__':
    main()
