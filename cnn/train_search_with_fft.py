import os
import sys
import time
import glob
import logging
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10

import utils
from model_search import FFTNetwork as Network
from genotypes import genomes
from architect import FFTArchitect as Architect

CIFAR_CLASSES = 10


def cmd_argument_parser():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data',
                        type=str,
                        default='../data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.025,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min',
                        type=float,
                        default=0.001,
                        help='min learning rate')
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
                        default=50,
                        help='num of training epochs')
    parser.add_argument('--init_channels',
                        type=int,
                        default=16,
                        help='num of init channels')
    parser.add_argument('--layers',
                        type=int,
                        default=8,
                        help='total number of layers')
    parser.add_argument('--model_path',
                        type=str,
                        default='saved_models',
                        help='path to save the model')
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
                        default=0.3,
                        help='drop path probability')
    parser.add_argument('--save',
                        type=str,
                        default='EXP',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip',
                        type=float,
                        default=5,
                        help='gradient clipping')
    parser.add_argument('--train_portion',
                        type=float,
                        default=0.5,
                        help='portion of training data')
    parser.add_argument('--unrolled',
                        action='store_true',
                        default=False,
                        help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate',
                        type=float,
                        default=3e-4,
                        help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay',
                        type=float,
                        default=1e-3,
                        help='weight decay for arch encoding')
    args = parser.parse_args()

    args.save = 'search_fft-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

    # The loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # Create the fixed network
    # Note: This differs from the Network used in model_search.py
    # TODO: Document the different constructors of Network
    model = Network(C=args.init_channels,
                    num_classes=CIFAR_CLASSES,
                    layers=args.layers,
                    criterion=criterion,
                    retain_arch_grad=True)
    model = model.cuda()

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # Optimizer used to adjust the models parameters as well as an optimizer
    # of the learning rate
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                     T_max=float(args.epochs),
                                     eta_min=args.learning_rate_min)

    # Create transforms
    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    # Get data for training
    # TODO: Might not be necessary for actual architect search with
    #       NASbench 301
    train_data = CIFAR10(root=args.data, train=True, download=True,
                         transform=train_transform)

    # Calculate split amounts
    n_train = len(train_data)
    indices = list(range(n_train))
    split = int(np.floor(n_train * args.train_portion))

    # Create data splits
    train_queue = DataLoader(train_data, batch_size=args.batch_size,
                             sampler=SubsetRandomSampler(indices[:split]),
                             pin_memory=True, num_workers=2)

    valid_queue = DataLoader(train_data, batch_size=args.batch_size,
                             sampler=SubsetRandomSampler(indices[split:]),
                             pin_memory=True, num_workers=2)

    # Create the architect
    architect = Architect(model, args)

    for epoch in range(args.epochs):
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(f'Normal alphas: {F.softmax(model.alphas_normal(), dim=-1)}')
        print(f'Reduce alphas: {F.softmax(model.alphas_reduce(), dim=-1)}')
        print(f'Normal softmax alphas: {F.softmax(model.alphas_normal(), dim=-1)}')
        print(f'Reduce softmax alphas: {F.softmax(model.alphas_reduce(), dim=-1)}')
        print(f'Normal cell coeffs: {model._coeffs["normal"]}' )
        print(f'Reduce cell coeffs: {model._coeffs["reduce"]}' )

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model,
                                     architect, criterion, optimizer, 
                                     lr_scheduler, args)

        # Validate model
        #with torch.no_grad():
            #valid_acc, valid_obj = infer(valid_queue, model, criterion, args)

        logging.info('train_acc %f', train_acc)
        #logging.info('valid_acc %f', valid_acc)

        # SAve the model for each epoch
        utils.save(model, os.path.join(args.save, 'weights.pt'))

        lr_scheduler.step()
        logging.info('epoch %d', epoch)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, 
          args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # Ensure model is put in training mode
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()

        # Get a random mini-batch from the validiation split
        # Loading it on to the device
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        # Step forward with the architecture
        architect.step(input, target, input_search, target_search, lr,
                       optimizer, unrolled=args.unrolled)

        # Get the loss of the model
        logits = model(input)
        loss = criterion(logits, target)

        # Optimize the model for a step
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

    # Ensure model is put into evaluation mode
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
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
