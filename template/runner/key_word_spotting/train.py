import time

import torch
import logging

from tqdm import tqdm

from template.runner.key_word_spotting.cosine_loss import CosineLoss
from util.misc import AverageMeter


def train(train_loader, model, optimizer, writer, epoch, no_cuda, log_interval, **kwargs):
    """
    Training routine

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.
    model : torch.nn.module
        The network model being used.
    optimizer : torch.optim
        The optimizer used to perform the weight update.
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes).
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    ----------
    int
        Placeholder 0. In the future this should become the FPR95
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)

    for batch_idx, (input, target) in pbar:

        loss = CosineLoss(size_average=False, use_sigmoid=True)

        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

    #     if train_loader_iter.batches_outstanding == 0:
    #         train_loader_iter = DataLoaderIter(loader=train_loader)
    #         logger.info('Resetting data loader')
    #     word_img, embedding, _, _ = train_loader_iter.next()
    #     if args.gpu_id is not None:
    #         if len(args.gpu_id) > 1:
    #             word_img = word_img.cuda()
    #             embedding = embedding.cuda()
    #         else:
    #             word_img = word_img.cuda(args.gpu_id[0])
    #             embedding = embedding.cuda(args.gpu_id[0])
    #
    #     word_img = torch.autograd.Variable(word_img)
    #     embedding = torch.autograd.Variable(embedding)
    #     output = cnn(word_img)
    #     ''' BCEloss ??? '''
    #     loss_val = loss(output, embedding) * args.batch_size
    #     loss_val.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    #
    # # mean runing errors??
    # if (epoch + 1) % args.display == 0:
    #     logging.info('Iteration %*d: %f', len(str(max_iters)), epoch + 1, loss_val.data[0])
    #
    # # change lr
    # if (epoch + 1) == args.learning_rate_step[lr_cnt][0] and (epoch + 1) != max_iters:
    #     lr_cnt += 1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.learning_rate_step[lr_cnt][1]

    return 0
