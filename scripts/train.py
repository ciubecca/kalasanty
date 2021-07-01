import argparse
import os
import time
import yaml
import pandas as pd
from kalasanty.data import DataWrapper
from kalasanty.net import UNet, dice_loss, dice, ovl

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def input_path(path):
    """Check if input exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' % path)
    return path


def output_path(path):
    path = os.path.abspath(path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', required=True, type=input_path,
                        help='path to the .hdf file with prepared data (can be '
                             'created with prepare_dataset.py)')
    parser.add_argument('--model', '-m', type=input_path,
                        help='path to the .hdf file with pretrained model. '
                             'If not specified, a new model will be trained from scratch.')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--steps_per_epoch', default=150, type=int)
    parser.add_argument('--epochs', default=10000, type=int)

    parser.add_argument("-r", "--runconfig", dest='runconfig', type=str, required=True,
                        help=f"The run config yaml file")

    parser.add_argument('--load', '-l', action='store_true',
                        help='whether to load all data into memory')
    parser.add_argument('--output', '-o', type=output_path,
                        help='name for the output directory. If not specified, '
                             '"output_<YYYY>-<MM>-<DD>" will be used')
    parser.add_argument('--verbose', '-v', default=2, type=int,
                        help='verbosity level for keras')
    parser.add_argument('--device', '-d', type=str, default='gpu', const='gpu', choices=['gpu', 'cpu'],
                        nargs='?', help='device')

    return parser.parse_args()


class LoadConfig:
    def __init__(self, path):
        runconfig = yaml.safe_load(open(path, 'r'))
        self.train_ids = runconfig['train']
        self.test_ids = runconfig['val']


def main():
    args = parse_args()

    runconfig = args.runconfig

    config = LoadConfig(runconfig)
    train_ids = config.train_ids
    test_ids = config.test_ids

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    if args.output is None:
        args.output = 'output_' + time.strftime('%Y-%m-%d')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.access(args.output, os.W_OK):
        raise IOError('Cannot create files inside %s (check your permissions).' % args.output)

    if train_ids:
        if test_ids:
            all_ids = sorted(set(train_ids) | set(test_ids))
        else:
            all_ids = train_ids
    else:
        all_ids = None

    data = DataWrapper(args.input, test_set=test_ids, pdbids=all_ids,
                       load_data=args.load)

    if args.model:
        model = UNet.load_model(args.model, data_handle=data)
    else:
        model = UNet(data_handle=data)
        model.compile(optimizer=Adam(lr=1e-6), loss=dice_loss,
                      metrics=[dice, ovl, 'binary_crossentropy'])

    train_batch_generator = data.batch_generator(batch_size=args.batch_size)

    callbacks = [ModelCheckpoint(os.path.join(args.output, 'checkpoint.hdf'),
                                 save_best_only=False)]

    if test_ids:
        val_batch_generator = data.batch_generator(batch_size=args.batch_size, subset='test')
        num_val_steps = max(args.steps_per_epoch // 5, 1)
        callbacks.append(ModelCheckpoint(os.path.join(args.output, 'best_weights.hdf'),
                                         save_best_only=True))
    else:
        val_batch_generator = None
        num_val_steps = None

    model.fit_generator(train_batch_generator, steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs, verbose=args.verbose, callbacks=callbacks,
                        validation_data=val_batch_generator, validation_steps=num_val_steps)

    history = pd.DataFrame(model.history.history)
    history.to_csv(os.path.join(args.output, 'history.csv'))
    model.save(os.path.join(args.output, 'model.hdf'))


if __name__ == '__main__':
    main()
