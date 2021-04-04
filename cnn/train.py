import os
import sys
import time
import glob
import logging
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10

import utils
from model import NetworkCIFAR as Network
from genotypes import genomes

CIFAR_CLASSES = 10


def cmd_argument_parser():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data',
                        type=str,
                        default='../data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.025,
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=3e-4,
                        help='weight decay')
    parser.add_argument('--report_freq',
                        type=float,
                        default=50,
                        help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs',
                        type=int,
                        default=600,
                        help='num of training epochs')
    parser.add_argument('--init_channels',
                        type=int,
                        default=36,
                        help='num of init channels')
    parser.add_argument('--layers',
                        type=int,
                        default=20,
                        help='total number of layers')
    parser.add_argument('--model_path',
                        type=str,
                        default='saved_models',
                        help='path to save the model')
    parser.add_argument('--auxiliary',
                        action='store_true',
                        default=False,
                        help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight',
                        type=float,
                        default=0.4,
                        help='weight for auxiliary loss')
    parser.add_argument('--cutout',
                        action='store_true',
                        default=False,
                        help='use cutout')
    parser.add_argument('--cutout_length',
                        type=int,
                        default=16,
                        help='cutout length')
    parser.add_argument('--drop_path_prob',
                        type=float,
                        default=0.2,
                        help='drop path probability')
    parser.add_argument('--save',
                        type=str,
                        default='EXP',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch',
                        type=str,
                        default='DARTS',
                        help='which architecture to use')
    parser.add_argument('--grad_clip',
                        type=float,
                        default=5,
                        help='gradient clipping')
    args = parser.parse_args()

    # TODO: Currently just using args to pass everything around
    #       probably better to update the parser arg name
    args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    # TODO: This should be outside of this function
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args


def create_logger(save_dir: str, filename: str = 'log.txt') -> None:
    """ Sets up the logger used

    Args:
        save_dir: directory to save to
        filename: the name of the log file
    """
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    formatter = logging.Formatter(log_format)

    save_path = os.path.join(save_dir, filename)
    fh = logging.FileHandler(save_path)
    fh.setFormatter(formatter)

    logging.getLogger().addHandler(fh)


def main():
    # Setup
    args = cmd_argument_parser()
    create_logger(args.save)

    # Set the gpu device to be used
    # NOTE: Only operates on a single GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    else:
        logging.info('no gpu device available')
        sys.exit(1)

    # Ensure seeds are set
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Hardware specific tuning
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Get the specific architecture to train
    genotype = genomes[args.arch]

    # Create the fixed network
    # Note: This differs from the Network used in model_search.py
    # TODO: Update the Network class
    model = Network(C=args.init_channels,
                    num_classes=CIFAR_CLASSES,
                    layers=args.layers,
                    auxiliary=args.auxiliary,
                    genotype=genotype)
    model = model.cuda()

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # The loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # Optimizer used to adjust the models parameters as well as an optimizer
    # of the learning rate
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                     T_max=float(args.epochs))

    # Get the transforms for both the train and validation data
    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # Get the data from torchvision's datasets
    train_data = CIFAR10(root=args.data, train=True, download=True,
                         transform=train_transform)
    valid_data = CIFAR10(root=args.data, train=False, download=True,
                         transform=valid_transform)

    # Create Dataloaders for both
    train_queue = DataLoader(train_data, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0)

    for epoch in range(args.epochs):
        logging.info(f'epoch = {epoch}')
        logging.info(f'lr = {lr_scheduler.get_last_lr()}')

        # More likely to drop a path as epochs progress
        model.drop_path_prob = args.drop_path_prob * (epoch / args.epochs)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, args)
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion, args)

        logging.info(f'train_acc = {train_acc}')
        logging.info(f'valid_acc = {valid_acc}')

        # Save the model for each epoch
        utils.save(model, os.path.join(args.save, 'weights.pt'))

        lr_scheduler.step()


def train(train_queue, model, criterion, optimizer, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # Ensure model is put in training mode
    # and grad calculations enabled
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # Load vectors onto device
        input = input.cuda()
        target = target.cuda()

        # Get the loss of the model, logits_aux used if auxiliary loss used
        logits, logits_aux = model(input)
        loss = criterion(logits, target)

        # If auxiliary loss should be used
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        # Optimization step on gradients
        # Extra clipping is performed to prevent and exploding gradient
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        # Record some metrics
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(f'train {step=} {objs.avg=} {top1.avg=} {top5.avg=}')

        return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # Ensure model is put in evaluation mode
    # and no grad caluclations made
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(f'train {step=} {objs.avg=} {top1.avg=} {top5.avg=}')

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
