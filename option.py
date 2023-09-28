import argparse

parser = argparse.ArgumentParser(description="DTI_ARB")
parser.add_argument("--block_size", type=tuple, default=(16,16,16),
                    help="Block Size")
parser.add_argument("--test_block_size", type=tuple, default=(16,16,16),
                    help="Block Size")
parser.add_argument("--crop_depth", type=int, default=15,
                    help="crop across z-axis")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int,
                    help="Batch_size")
parser.add_argument("--sort", type=bool,
                    help="Sort Subject Ids")
parser.add_argument("--debug", type=bool,
                    help="Print additional input")
parser.add_argument("--preload", type=bool,
                    help="Preload data into memory")
parser.add_argument("--ret_points", type=bool, default=False,
                    help="return box point of crops")
parser.add_argument("--thres", type=float, default=0.6,
                    help="threshold for blk emptiness")
parser.add_argument("--offset", type=int, default=20,
                    help="epoch with scale (1,1,1)")
parser.add_argument("--gaps", type=int, default=20,
                    help="number of epochs of gap between each scale change")
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Log specifications
parser.add_argument('--save', type=str, default='DTIArbNet',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
args = parser.parse_args()
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 16
args.sort = True
args.cuda = True
args.scale = (1,1,1)
args.epochs = 400
args.gaps = 20
args.offset = 10