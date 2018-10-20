import logging
import numpy as np

# DeepDIVA
import models

from template.setup import set_up_model
from template.runner.key_word_spotting.setup import setup_dataloaders
from template.runner.key_word_spotting import train


class KeyWordSpotting:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr, validation_interval, batch_size,
                   **kwargs):
        """
                This is the main routine where train(), validate() and test() are called.

        """

        # Setting up the dataloaders
        train_loader, test_loader, num_classes = setup_dataloaders(batch_size=1, **kwargs)

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr,
                                                                            train_loader=train_loader,
                                                                            **kwargs)


        # Core routine
        logging.info('Begin training')
        val_value = np.zeros((epochs - start_epoch))
        train_value = np.zeros((epochs - start_epoch))


        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = KeyWordSpotting._train(train_loader=test_loader,
                                                        model=model,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        writer=writer,
                                                        epoch=epoch,
                                                        batch_size=batch_size,
                                                        **kwargs)
            # Validate
            if epoch % validation_interval == 0:
                val_value[epoch] = KeyWordSpotting._validate(test_loader, model, None, writer, -1, **kwargs)

        # Test
        logging.info('Training completed')

        test_value = KeyWordSpotting._test(test_loader=test_loader,
                                           model=model,
                                           criterion=criterion,
                                           writer=writer,
                                           epoch=(epochs - 1),
                                           **kwargs)

        return train_value, val_value, test_value

    ####################################################################################################################
    # These methods delegate their function to other classes in this package.
    # It is useful because sub-classes can selectively change the logic of certain parts only.

    @classmethod
    def _train(cls, train_loader, model, optimizer, batch_size, **kwargs):
        return train.train(train_loader, model, optimizer, batch_size, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, **kwargs)

        # logging.info('Training:')
        # for iter_idx in range(start_epoch, ):
        #     if iter_idx % args.test_interval == 0:  # and iter_idx > 0:
        #         logging.info('Evaluating net after %d iterations', iter_idx)
        #         evaluate_cnn(cnn=cnn,
        #                      dataset_loader=test_loader,
        #                      args=args)
        #     for _ in range(args.iter_size):
        #         if train_loader_iter.batches_outstanding == 0:
        #             train_loader_iter = DataLoaderIter(loader=train_loader)
        #             logging.info('Resetting data loader')
        #         word_img, embedding, _, _ = train_loader_iter.next()
        #         if args.gpu_id is not None:
        #             if len(args.gpu_id) > 1:
        #                 word_img = word_img.cuda()
        #                 embedding = embedding.cuda()
        #             else:
        #                 word_img = word_img.cuda(args.gpu_id[0])
        #                 embedding = embedding.cuda(args.gpu_id[0])
        #
        #         word_img = torch.autograd.Variable(word_img)
        #         embedding = torch.autograd.Variable(embedding)
        #         output = cnn(word_img)
        #         ''' BCEloss ??? '''
        #         loss_val = loss(output, embedding) * args.batch_size
        #         loss_val.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #
        #     # mean runing errors??
        #     if (iter_idx + 1) % args.display == 0:
        #         logging.info('Iteration %*d: %f', len(str(max_iters)), iter_idx + 1, loss_val.data[0])
        #
        #     # change lr
        #     if (iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx + 1) != max_iters:
        #         lr_cnt += 1
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = args.learning_rate_step[lr_cnt][1]
        #
        #     # if (iter_idx + 1) % 10000 == 0:
        #     #    torch.save(cnn.state_dict(), 'PHOCNet.pt')
        #     # .. to load your previously training model:
        #     # cnn.load_state_dict(torch.load('PHOCNet.pt'))
