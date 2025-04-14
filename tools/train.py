
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import _init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.function import train
from lib.core.function import validate
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary
from lib.models import pose_hrnet_part
import lib.models 
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

    # philly
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

    parser.add_argument('--fewshot',
                        help='train on ap10k with few shot annotations',
                        action='store_true')

    parser.add_argument('--pretrained',
                        help='path for pretrained model',
                        type=str,
                        default='')
    parser.add_argument('--resume', help='path to resume', type=str, default='')

    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(model, dump_input))

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    # change mean nad std accordingly
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.animalpose:
        train_dataset = eval('dataset_animal.' + cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = eval('dataset_animal.' + 'ap10k')(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    last_epoch = -1
    best_perf_epoch = 0
    optimizer = get_optimizer(cfg, model)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    # load pretrained model
    if os.path.exists(args.pretrained):
        logger.info("=> loading checkpoint '{}'".format(args.pretrained))
        pretrained_model = torch.load(args.pretrained)
        model.load_state_dict(pretrained_model['state_dict'])

    # resume pretrained model
    if os.path.exists(args.resume):
        logger.info("=> resume from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # evaluate
    if args.evaluate:

        acc = validate(cfg, valid_loader, valid_dataset, model, criterion,
                        final_output_dir, tb_log_dir, writer_dict, args.animalpose)
        return

    # # train
    # for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
    #     lr_scheduler.step()

    #     # train for one epoch
    #     train(cfg, train_loader, model, criterion, optimizer, epoch,
    #           final_output_dir, tb_log_dir, writer_dict)

    #     # evaluate on validation set
    #     perf_indicator = validate(
    #         cfg, valid_loader, valid_dataset, model, criterion,
    #         final_output_dir, tb_log_dir, writer_dict, args.animalpose
    #     )

    #     if perf_indicator >= best_perf:
    #         best_perf = perf_indicator
    #         best_model = True
    #         best_perf_epoch = epoch + 1
    #     else:
    #         best_model = False

    #     logger.info('=> saving checkpoint to {}'.format(final_output_dir))
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'model': cfg.MODEL.NAME,
    #         'state_dict': model.module.state_dict(),
    #         'perf': perf_indicator,
    #         'optimizer': optimizer.state_dict(),
    #     }, best_model, final_output_dir)

    # final_model_state_file = os.path.join(
    #     final_output_dir, 'final_state.pth'
    # )
    # logger.info('=> saving final model state to {}'.format(
    #     final_model_state_file)
    # )
    # logger.info('Best accuracy {} at epoch {}'.format(best_perf, best_perf_epoch))
    # torch.save(model.module.state_dict(), final_model_state_file)
    # writer_dict['writer'].close()

    # train
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        
        # Step optimizer first
        optimizer.step()
        
        # Then update learning rate
        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict, args.animalpose
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
            best_perf_epoch = epoch + 1
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)



if __name__ == '__main__':
    main()
