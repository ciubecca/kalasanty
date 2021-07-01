import argparse
import os
import re
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


from pybel import readfile
from kalasanty.net import UNet
from tfbio.data import Featurizer

from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(root_dir, 'data', 'model_scpdb2017.hdf')


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

    parser.add_argument('--input', '-i', required=True, type=input_path, nargs='+',
                        help='paths to protein structures')
    parser.add_argument('--model', '-m', type=input_path, default=model_path,
                        help='path to the .hdf file with trained model.')
    parser.add_argument('--dirname_pattern', type=str,
                        default='.*{sep}([^{sep}]+){sep}[^{sep}]+$'.format(sep=os.sep),
                        help='pattern to extract molecule name from path')
    parser.add_argument('--output', '-o', type=output_path,
                        help='name for the output directory. If not specified, '
                             '"pockets_<YYYY>-<MM>-<DD>" will be used')
    parser.add_argument('--format', '-f', default='mol2',
                        help='input format; can be any format for 3D structures'
                             ' supported by Open Babel')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='whether to print messages')
    parser.add_argument('--max_dist', type=float, default=35,
                        help='max_dist parameter used for training set')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scale parameter used for training set')
    parser.add_argument('--device', '-d', type=str, default='gpu', const='gpu', choices=['gpu', 'cpu'],
                        nargs='?', help='device')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    if args.output is None:
        args.output = 'pockets_' + time.strftime('%Y-%m-%d')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.access(args.output, os.W_OK):
        raise IOError('Cannot create files inside %s (check your permissions).' % args.output)

    # load trained model
    model = UNet.load_model(args.model, scale=args.scale, max_dist=args.max_dist,
                            featurizer=Featurizer(save_molecule_codes=False))

    if args.verbose:
        progress_bar = tqdm
    else:
        progress_bar = iter

    for path in progress_bar(args.input):
        match = re.match(args.dirname_pattern, path)
        if not match:
            raise ValueError('Cannot extract name from %s. '
                             'Please specify correct --namedir_pattern' % path)
        dirname = os.path.join(args.output, match.groups()[0])
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        mol = next(readfile(args.format, path))

        # predict pockets and save them as separate mol2 files
        pockets = model.predict_pocket_atoms(mol)
        for i, pocket in enumerate(pockets):
            pocket.write('mol2', os.path.join(dirname, 'pocket%i.mol2' % i), overwrite=True)

        # save pocket probability as density map (UCSF Chimera format)
        density, origin, step = model.pocket_density_from_mol(mol)
        model.save_density_as_cmap(density, origin, step, fname=os.path.join(dirname, 'pockets.cmap'))


if __name__ == '__main__':
    main()
