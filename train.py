#!/usr/bin/env python
import os, argparse, time
import utils
import torch
from torch import nn, optim 

OUTPUT_SIZE = 129 # 0-127 notes + 1 for rests

def parse_args():

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.add_argument('--experiment_dir', type=str,
                        default='experiments/default',
                        help='directory to store checkpointed models and tensorboard logs.' \
                             'if omitted, will create a new numbered folder in experiments/.')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate. If not specified, the recommended learning '\
                        'rate for the chosen optimizer is used.')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for RNN input per step.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs before stopping training.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='percentage of weights that are turned off every training '\
                        'set step. This is a popular regularization that can help with '\
                        'overfitting. Recommended values are 0.2-0.5')
    parser.add_argument('--optimizer', 
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 
                                 'adam', 'adamax', 'nadam'], default='adam',
                        help='The optimization algorithm to use. '\
                        'See https://keras.io/optimizers for a full list of optimizers.')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value.')
    parser.add_argument('--message', '-m', type=str,
                        help='a note to self about the experiment saved to message.txt '\
                        'in --experiment_dir.')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, 
                        help='Number of CPUs to use when loading and parsing midi files.')
    parser.add_argument('--max_files_in_ram', default=25, type=int,
                        help='The maximum number of midi files to load into RAM at once.'\
                        ' A higher value trains faster but uses more RAM. A lower value '\
                        'uses less RAM but takes significantly longer to train.')
    return parser.parse_args()

# create or load a saved model
# returns the model and the epoch number (>1 if loaded from checkpoint)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=OUTPUT_SIZE,
                            hidden_size=args.rnn_size,
                            num_layers=args.num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.rnn_size, OUTPUT_SIZE)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)

def get_model(args, experiment_dir=None):
    
    epoch = 0
    
    if not experiment_dir:
        model = Model(args)

    else:
        model, epoch = utils.load_model_from_checkpoint(experiment_dir)

    # these cli args aren't specified if get_model() is being
    # being called from sample.py
    # ----------------------------------------------------------


    # model.compile(loss='categorical_crossentropy', 
    #               optimizer=optimizer,
    #               metrics=['accuracy'])
    return model, epoch

def get_optimizer(args, model):
    if args.learning_rate == None:
        utils.log(
            'Error: Please define --learning_rate. Exiting.',
            True)
        exit(1)
    if 'optimizer' in args: # 'grad_clip' in args and
        #kwargs = { 'clipvalue': args.grad_clip }

        # select the optimizers
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), args.learning_rate)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), args.learning_rate)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), args.learning_rate)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), args.learning_rate)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), args.learning_rate)
        elif args.optimizer == 'adamax':
            optimizer = optim.Adamax(model.parameters(), args.learning_rate)
        else:
            utils.log(
                'Error: {} is not a supported optimizer. Exiting.'.format(args.optimizer),
                True)
            exit(1)
    else: # so instead lets use a default (no training occurs anyway)
        optimizer = optim.Adam(model.parameters(), args.learning_rate)
    return optimizer

def get_callbacks(experiment_dir, checkpoint_monitor='val_acc'):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(experiment_dir, 'tensorboard-logs'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))

    return callbacks

def main():

    args = parse_args()
    args.verbose = True

    try:
        # get paths to midi files in --data_dir
        midi_files = [os.path.join(args.data_dir, path) \
                      for path in os.listdir(args.data_dir) \
                      if '.mid' in path or '.midi' in path]
    except OSError as e:
        log('Error: Invalid --data_dir, {} directory does not exist. Exiting.', args.verbose)
        exit(1)

    utils.log(
        'Found {} midi files in {}'.format(len(midi_files), args.data_dir),
        args.verbose
    )

    if len(midi_files) < 1:
        utils.log(
            'Error: no midi files found in {}. Exiting.'.format(args.data_dir),
            args.verbose
        )
        exit(1)

    # create the experiment directory and return its name
    experiment_dir = utils.create_experiment_dir(args.experiment_dir, args.verbose)

    # write --message to experiment_dir
    if args.message:
        with open(os.path.join(experiment_dir, 'message.txt'), 'w') as f:
            f.write(args.message)
            utils.log('Wrote {} bytes to {}'.format(len(args.message), 
                os.path.join(experiment_dir, 'message.txt')), args.verbose)

    val_split = 0.2 # use 20 percent for validation
    val_split_index = int(float(len(midi_files)) * val_split)

    # use generators to lazy load train/validation data, ensuring that the
    # user doesn't have to load all midi files into RAM at once
    train_generator = utils.get_data_generator(midi_files[0:val_split_index], 
                                               window_size=args.window_size,
                                               batch_size=args.batch_size,
                                               num_threads=args.n_jobs,
                                               max_files_in_ram=args.max_files_in_ram)

    val_generator = utils.get_data_generator(midi_files[val_split_index:], 
                                             window_size=args.window_size,
                                             batch_size=args.batch_size,
                                             num_threads=args.n_jobs,
                                             max_files_in_ram=args.max_files_in_ram)
train.py --niter 20 --niter_decay 20 --save_epoch_freq 10 --dataset_mode npy_aligned_3d --dataroot ../3D_460_patchified_norm/ --model paired_revgan3d --name 3d_460 --which_model_netG edsrF_generator_3d --gpu_ids 0,1 --batchSize 2 --which_model_netD n_layers --n_layers_D 2 --lr_G 0.0001 --lr_D 0.0004
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print(device)

    model, epoch = get_model(args)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if args.verbose:
        print(model)

    utils.save_model(model, experiment_dir)
    utils.log('Saved model to {}'.format(os.path.join(experiment_dir, 'model.pth')),
              args.verbose)

    #callbacks = get_callbacks(experiment_dir)

    criterion = nn.NLLLoss().float()
    optimizer = get_optimizer(args, model)
    
    print('fitting model...')
    start_time = time.time()

    train_losses, val_losses = [], []
    for e in range(args.num_epochs):
        print('Epoch', e+1)
        running_loss = 0
        len_train_generator = 0
        len_val_generator = 0
        for x, y in train_generator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            log_ps = model(x.float())
            loss = criterion(log_ps, y.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            len_train_generator += 1
            #print(len_train_generator)
        else:
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for x, y in val_generator:
                    x, y = x.to(device), y.to(device)
                    log_ps = model(x.float())
                    val_loss += criterion(log_ps, y.long())
                    # convert log probabilities to probabilites
                    ps = torch.exp(log_ps)
                    # select the highest
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    len_val_generator += 1
        # set the model back to train mode after the eval mode
        model.train()
        
        train_losses.append(running_loss/len_train_generator)
        val_losses.append(val_loss/len_val_generator)
        
        print("\nEpoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}   ".format(running_loss/len_train_generator),
              "Validation Loss: {:.3f}   ".format(val_loss/len_val_generator),
              "Validation Accuracy: {:.3f}".format(accuracy/len_val_generator))
        
        utils.save_model(model, experiment_dir)
        # Model Checkpoint
        # if val_losses[-1] < best_val_loss:
        #     print('Validation loss improved from {:.3f} to {:.3f}, saving the model.'.format(best_val_loss,
        #                                                                                      val_losses[-1]))
        #     best_val_loss = val_losses[-1]
        #     checkpoint = {'model': model,
        #                   'idx_to_class': idx_to_class}
        #     torch.save(checkpoint, 'checkpoint.pth')
 

    utils.log('Finished in {:.2f} seconds'.format(time.time() - start_time), args.verbose)

if __name__ == '__main__':
    main()