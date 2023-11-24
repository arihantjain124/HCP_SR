import argparse

parser = argparse.ArgumentParser(description="DTI_ARB")
parser.add_argument("--block_size", type=tuple, default=(16,16,4),
                    help="Block Size")
parser.add_argument("--test_block_size", type=tuple, default=(16,16,4),
                    help="Block Size")
parser.add_argument("--stride", type=tuple, default=(0,0,0),
                    help="Testing Dataset Stride")
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
parser.add_argument("--gap", type=int, default=20,
                    help="number of epochs of gap between each scale change")

parser.add_argument("--no_vols", type=int, default=20,
                    help="Number of Volumes to load")


# Optimization specifications
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=40,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')

# Loss specifications
parser.add_argument('--loss', type=str, default='0.5*MSE+0.5*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')


# Log specifications
parser.add_argument('--run_name', type=str, default='secondrun',
                    help='file name to save')
parser.add_argument('--save', type=str, default='DTIArbNet',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')

parser.add_argument('--print_every', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_every', type=int, default=30,
                    help='how many batches to wait before logging training status')



# Hardware specifications
# parser.add_argument('--n_threads', type=int, default=2,
#                     help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False,
                    help='use cpu only')
parser.add_argument('--gpu', type=int, default=0,
                    help='use cpu only')
# parser.add_argument('--n_GPUs', type=int, default=2,
#                     help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--pin_mem', action='store_true',
                    help='pin memory for dataloader')
parser.add_argument("--train_set", type=float, default=0.7,
                    help="percentage of data to be used for training")


# Model specifications
parser.add_argument('--model', default='dmri_arb',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default= 'None',
                    help='pre-trained model directory')
# parser.add_argument('--extend', type=str, default='.',
#                     help='pre-trained model directory')
# parser.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')
# parser.add_argument('--shift_mean', default=True,
#                     help='subtract pixel mean from the input')
# parser.add_argument('--dilation', action='store_true',
#                     help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')


args = parser.parse_args()
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 16
args.sort = True
args.cuda = True
args.scale = (1,1,1)
args.epochs = 400
args.offset = 2