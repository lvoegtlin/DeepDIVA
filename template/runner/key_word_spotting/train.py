import time

import torch
import logging
import numpy as np

from torch.utils.data.dataloader import DataLoaderIter
from tqdm import tqdm

from template.runner.key_word_spotting.cosine_loss import CosineLoss
from template.runner.key_word_spotting.retrieval import map_from_query_test_feature_matrices


def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]


def train(test_loader, model, optimizer, batch_size, iter_size, learning_rate_step, test_interval, gpu_id,
          **kwargs):
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

    lr_cnt = 0
    max_iters = learning_rate_step[-1][0]

    loss = CosineLoss(size_average=False, use_sigmoid=True)
    optimizer.zero_grad()

    logging.info('Training:')
    for iter_idx in range(max_iters):
        if iter_idx % test_interval == 0:  # and iter_idx > 0:
            logging.info('Evaluating net after %d iterations', iter_idx)
            evaluate_cnn(cnn=model,
                         dataset_loader=test_loader,
                         gpu_id=gpu_id)
            for _ in range(iter_size):
                if train_loader_iter.batches_outstanding == 0:
                    train_loader_iter = DataLoaderIter(loader=test_loader)
                    logging.info('Resetting data loader')
                word_img, embedding, _, _ = train_loader_iter.next()
                if gpu_id is None:
                    word_img = word_img.cuda()
                    embedding = embedding.cuda()

                word_img = torch.autograd.Variable(word_img)
                embedding = torch.autograd.Variable(embedding)
                output = model(word_img)
                ''' BCEloss ??? '''
                loss_val = loss(output, embedding) * batch_size
                loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            # mean runing errors??
            if (iter_idx + 1) % 500 == 0:
                logging.info('Iteration %*d: %f', len(str(max_iters)), iter_idx + 1, loss_val.data[0])

            # change lr
            if (iter_idx + 1) == learning_rate_step[lr_cnt][0] and (iter_idx + 1) != max_iters:
                lr_cnt += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate_step[lr_cnt][1]


def evaluate_cnn(cnn, dataset_loader, gpu_id):
    logger = logging.getLogger('PHOCNet-Experiment::test')
    # set the CNN in eval mode
    cnn.eval()
    logger.info('Computing net output:')
    qry_ids = []
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)

    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm(dataset_loader)):
        if gpu_id is None:
            # in one gpu!!
            word_img = word_img.cuda(async=True)
            embedding = embedding.cuda(async=True)
            # word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''
        output = torch.sigmoid(cnn(word_img))
        # output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0, 0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  # [sample_idx] = is_query[0]

    '''
    # find queries

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed

    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                         test_features=outputs,
                                                         query_labels=qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[0] + ave_precs_qbe[1]) * 100)

    # clean up -> set CNN in train mode again
    cnn.train()


