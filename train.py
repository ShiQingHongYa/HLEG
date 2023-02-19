import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

import _init_paths
# from dataset.get_dataset import get_datasets

from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict

from data_utils.get_dataset_new import get_datasets, CLASS_9, CLASS_15
from data_utils.metrics import validate_f1


def parser_args():
    available_models = ['Q2L-R101-448']
    parser = argparse.ArgumentParser(description='HLEG Intention Training')
    parser.add_argument('--dataname', help='dataname', default='intentonomy', choices=['intentonomy'])
    # YOUR_PATH/DataSet/Intentonomy 
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/data/sqhy_data/intent_resize/low')
    parser.add_argument('--img_size', default=224, type=int,help='size of input images')

    parser.add_argument('--output', metavar='DIR', default='/data/sqhy_model/HLEG',
                        help='path to output folder')
    parser.add_argument('--num_class', default=28, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +' | '.join(available_models) +
                        ' (default: Q2L-R101-448)')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=True,
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=6, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://127.0.0.1:3717', type=str,
    #                     help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=31, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args



best_mAP = 0
best_f1_samples = 0

def main():
    args = get_args()
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    # if args.seed is not None:
    #     torch.manual_seed(args.seed)  # Current CPU
    #     np.random.seed(args.seed)  # Numpy module
    #     random.seed(args.seed)  # Python random module
    #     torch.backends.cudnn.deterministic = True # Close optimization
    #     torch.backends.cudnn.benchmark = False  # Close optimization
    #     torch.cuda.manual_seed(args.seed)  # Current GPU
    #     torch.cuda.manual_seed_all(args.seed)  # All GPU (Optional)
    seed_everything(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "_config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP
    global best_f1_samples

    # build model
    model = build_q2l(args)

    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False,find_unused_parameters=True)

    # criterion
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )          
    else:
        raise NotImplementedError


    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    if args.evaluate:
        _, mAP, f1_micros, f1_macros, f1_samples = validate(val_loader, model, criterion, args, logger)
        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        logger.info('Validation: f1_micros: {}, f1_macros: {}, f1_samples: {}'.format(f1_micros, f1_macros, f1_samples))
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    f1s = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    f1s_ema = AverageMeter('mAP', ':5.5f', val_only=True)

    f1_micros = AverageMeter('f1_micro', ':5.5f', val_only=True)
    f1_macros = AverageMeter('f1_macro', ':5.5f', val_only=True)
    f1_samples = AverageMeter('f1_samples', ':5.5f', val_only=True)

    f1_micros_ema = AverageMeter('f1_micro_ema', ':5.5f', val_only=True)
    f1_macros_ema = AverageMeter('f1_macro_ema', ':5.5f', val_only=True)
    f1_samples_ema = AverageMeter('f1_samples_ema', ':5.5f', val_only=True)
    
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, f1_micros, f1_macros, f1_samples, losses_ema, f1_micros_ema, f1_macros_ema, f1_samples_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_epoch = -1
    # samples F1 max
    best_regular_f1 = 0
    best_ema_f1 = 0
    regular_f1_list = []
    ema_f1_list = []
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        start_epoch = time.time()
        loss, fine_l, middle_l, coarse_l = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)
        end_epoch = time.time()
        time_epoch = end_epoch - start_epoch
        print("the time of one epoch:", time_epoch)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('fine_loss', np.mean(np.array(fine_l)), epoch)
            summary_writer.add_scalar('middle_loss', np.mean(np.array(middle_l)), epoch)
            summary_writer.add_scalar('coarse_loss', np.mean(np.array(coarse_l)),epoch)
            # summary_writer.add_scalar('train_acc1', acc1, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            loss, mAP, f1_micros, f1_macros, f1_samples= validate(val_loader, model, criterion, args, logger)
            loss_ema, mAP_ema, f1_micros_ema, f1_macros_ema, f1_samples_ema = validate(val_loader, ema_m.module, criterion, args, logger)

            losses.update(loss)
            mAPs.update(mAP)
            f1s.update(f1_samples)
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)
            f1s_ema.update(f1_samples_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_f1_list.append(f1_samples)
            ema_f1_list.append(f1_samples_ema)

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_f1_micros', f1_micros, epoch)
                summary_writer.add_scalar('val_f1_macros', f1_macros, epoch)
                summary_writer.add_scalar('val_f1_samples', f1_samples, epoch)
                
                summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)
                summary_writer.add_scalar('val_f1_micros_ema', f1_micros_ema, epoch)
                summary_writer.add_scalar('val_f1_macros_ema', f1_macros_ema, epoch)
                summary_writer.add_scalar('val_f1_samples_ema', f1_samples_ema, epoch)

            # remember best (regular) F1_samples and corresponding epochs
            if f1_samples > best_regular_f1:
                best_regular_f1 = max(best_regular_f1, f1_samples)
                best_regular_epoch = epoch
            if f1_samples_ema > best_ema_f1:
                best_ema_f1 = max(f1_samples_ema, best_ema_f1)

            if f1_samples_ema > f1_samples:
                f1_samples = f1_samples_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = f1_samples > best_f1_samples
            if is_best:
                best_epoch = epoch
            best_f1_samples = max(f1_samples, best_f1_samples)

            logger.info("{} | Set best f1 {} in ep {}".format(epoch, best_f1_samples, best_epoch))
            logger.info("   | best regular f1 {} in ep {}".format(best_regular_f1, best_regular_epoch))

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'best_f1_samples': best_f1_samples,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best,filename=os.path.join(args.output, 'checkpoint.pth.tar'))
            # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))

            if math.isnan(loss) or math.isnan(loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_f1_samples': best_f1_samples,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best,filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_f1_list) > 1 and ema_f1_list[-1] < best_ema_f1:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                        break

    print("Best f1_samples:", best_f1_samples)

    if summary_writer:
        summary_writer.close()
    return 0



def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    end = time.time()
    fine_loss = []
    middle_loss = []
    coarse_loss = []

    for i, (images, target) in enumerate(stable(train_loader, args.seed+epoch)):

        # measure data loading time
        data_time.update(time.time() - end)

        # hierarchical category tree
        if args.dataname == "intentonomy":
            labels = np.array(target)
            labels_9 = []
            labels_15 = []
            for k in range(len(labels)):
                index_28 = [t for t, j in enumerate(labels[k]) if j == 1]
                index_15 = [t for t in CLASS_15 for k, m in enumerate(index_28) if m in set(CLASS_15[t])]
                label_15 = [0 for t in range(15)]
                for t in index_15:
                    if label_15[int(t)] != 1:
                        label_15[int(t)] = 1.0
                labels_15.append(label_15)

                index_9 = [t for t in CLASS_9 for k, m in enumerate(index_15) if int(m) in set(CLASS_9[t])]
                label_9 = [0 for t in range(9)]
                for t in index_9:
                    if label_9[int(t)] != 1:
                        label_9[int(t)] = 1.0
                labels_9.append(label_9)

            labels_15 = torch.from_numpy(np.array(labels_15))
            labels_9 = torch.from_numpy(np.array(labels_9))

            labels_9 = labels_9.cuda(non_blocking=True)
            labels_15 = labels_15.cuda(non_blocking=True)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss_fine = criterion(output[0], target)
            loss_middle = criterion(output[1], labels_15)
            loss_coarse = criterion(output[2], labels_9)
            com_loss = compare_loss(loss_fine, loss_middle, loss_coarse)
            loss_fine1 = loss_fine.sum() / loss_fine.size(0)
            loss = com_loss + 0.1 * loss_fine1

            if args.loss_dev > 0:
                loss *= args.loss_dev

        # record loss
        fine_l = loss_fine.unsqueeze(1).sum() / loss_fine.size(0)
        fine_loss.append(fine_l.item())
        middle_l = loss_middle.unsqueeze(1).sum()/loss_middle.size(0)
        middle_loss.append(middle_l.item())
        coarse_l = loss_coarse.unsqueeze(1).sum()/loss_coarse.size(0)
        coarse_loss.append(coarse_l.item())

        losses.update(loss.item(), images.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        if epoch >= args.ema_epoch:
            ema_m.update(model)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg, fine_loss, middle_loss, coarse_loss



@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    # mAP = AverageMeter('mAP', ':5.3f', val_only=)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                loss = criterion(output[0], target)
                loss = loss.sum()/loss.size(0)
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                output_sm = nn.functional.sigmoid(output[0])
                if torch.isnan(loss):
                    saveflag = True

            # record loss
            losses.update(loss.item(), images.size(0))
            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            saved_data.append(_item)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )
        
        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating metrics:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP                
            mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            metric_f1 = validate_f1
            f1_dict = metric_f1([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class)
            f1_micros = f1_dict['val_micro']
            f1_macros = f1_dict['val_macro']
            f1_samples = f1_dict['val_samples']
            
            logger.info("  mAP: {}".format(mAP))
            logger.info("  aps: {}".format(np.array2string(aps, precision=5)))
            logger.info("  f1_micros: {}".format(np.array2string(f1_micros, precision=5)))
            logger.info("  f1_macros: {}".format(np.array2string(f1_macros, precision=5)))
            logger.info("  f1_samples: {}".format(np.array2string(f1_samples, precision=5)))
            
        else:
            mAP = 0
            f1_micros = 0
            f1_macros = 0
            f1_samples = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP, f1_micros, f1_macros, f1_samples


##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] +'/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
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
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                             val=str(datetime.timedelta(seconds=int(self.val))), 
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

def compare_loss(loss_fine, loss_middle, loss_coarse):
    loss_fine = loss_fine.unsqueeze(1)
    loss_middle = loss_middle.unsqueeze(1)
    loss_coarse = loss_coarse.unsqueeze(1)
    w = torch.cat((loss_fine, loss_middle, loss_coarse), dim=1)
    max_w = torch.max(w, dim=1)
    loss = max_w[0].sum() / loss_fine.size(0)
    return loss

def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)       # Current CPU
        torch.cuda.manual_seed(seed)  # Current GPU
        np.random.seed(seed)          # Numpy module
        random.seed(seed)             # Python random module
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False    # Close optimization
        torch.backends.cudnn.deterministic = True # Close optimization
        torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader

if __name__ == '__main__':

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '3718'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    # os.environ['WORLD_SIZE'] = '2'
    main()

