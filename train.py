import time

from models import CycleGANModel
from utils import UnalignedDataset, TrainOptions

if __name__ == '__main__':

    opt = TrainOptions().parse()  # get training options

    dataset = UnalignedDataset(opt)

    model = CycleGANModel(opt)
    model.setup(opt)

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
