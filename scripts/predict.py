import argparse
import os
import re
import sys
import time
import yaml
from pathlib import  Path

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

    parser.add_argument('--destDataset', required=True, type=input_path,
                        help='path to the processed database')

    parser.add_argument('--model', '-m', type=input_path, default=model_path,
                        help='path to the .hdf file with trained model.')
    parser.add_argument('--dirname_pattern', type=str,
                        default='.*{sep}([^{sep}]+){sep}[^{sep}]+$'.format(sep=os.sep),
                        help='pattern to extract molecule name from path')
    parser.add_argument('--output', '-o', type=output_path,
                        help='name for the output directory. If not specified, '
                             '"pockets_<YYYY>-<MM>-<DD>" will be used')
    parser.add_argument('--format', '-f', default='pdb',
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

    parser.add_argument("-r", "--runconfig", dest='runconfig', type=str, required=True,
                        help=f"The run config yaml file")

    return parser.parse_args()

class LoadConfig:
    def __init__(self, path):
        runconfig = yaml.safe_load(open(path, 'r'))
        self.test_ids = runconfig['test']


def main():
    args = parse_args()

    runconfig = args.runconfig

    config = LoadConfig(runconfig)
    test_ids = config.test_ids

    destData = args.destDataset
    if not os.path.exists(destData):
        print(f"{destData} does not exist.", file=sys.stderr)
        sys.exit(-1)

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

    for test_id in test_ids:

        path = str(Path(destData) / test_id / f"{test_id}_selected.pdb")
        mol = next(readfile(args.format, path))

        # predict pockets and save them as separate mol2 files
        pockets = model.predict_pocket_atoms(mol)
        for i, pocket in enumerate(pockets):
            pocket.write('pdb', os.path.join(args.output, f'{test_id}_predictions.pdb'), overwrite=True)

        # save pocket probability as density map (UCSF Chimera format)
        density, origin, step = model.pocket_density_from_mol(mol)
        model.save_density_as_cmap(density, origin, step, fname=os.path.join(args.output, 'pockets.cmap'))


if __name__ == '__main__':
    main()
