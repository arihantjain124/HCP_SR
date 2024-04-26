import argparse

parser = argparse.ArgumentParser(description="DTI_ARB")
parser.add_argument("--block_size", type=tuple, default=(32,32,4),
                    help="Block Size")
parser.add_argument("--var_blk_size", type=bool, default=True,
                    help="Block Size")
# parser.add_argument("--enc", type=str, default='rdn',
#                     help="Encoder Type")
parser.add_argument("--start_var", type=bool, default=True,
                    help="Block Size")
parser.add_argument("--epochs", type=int, default=100,
                    help="Epochs")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int , default= 16,
                    help="Batch_size")
parser.add_argument("--test_batch_size", type=int , default= 8,
                    help="Batch_size")
parser.add_argument("--sort", type=bool,default=True,
                    help="Sort Subject Ids")
parser.add_argument("--debug", type=bool,
                    help="Print additional input")
parser.add_argument("--preload", type=bool,
                    help="Preload data into memory")
parser.add_argument("--ret_points", type=bool, default=False,
                    help="return box point of crops")
parser.add_argument("--enable_thres", type=bool, default=True,
                    help="threshold")
# parser.add_argument("--test_mask", type=bool, default=True,
#                     help="threshold")
parser.add_argument("--thres", type=float, default=0.3,
                    help="threshold for blk emptiness")

parser.add_argument("--no_vols", type=int, default=20,
                    help="Number of Volumes to load")
parser.add_argument("--test_vols", type=int, default=20,
                    help="Number of Volumes to load")

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--max_lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=18,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.8,
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
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')


# Log specifications
parser.add_argument('--run_name', type=str, default='..',
                    help='file name to save')
parser.add_argument('--save', type=str, default='DTIArb',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--save_models',  type=bool , default=False,
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
# parser.add_argument("--train_set", type=float, default=0.7,
#                     help="percentage of data to be used for training")


# Model specifications
parser.add_argument('--model', default='dmri_rdn',
                    help='model name')
parser.add_argument('--encoder', default='rdb',
                    help='model name')
parser.add_argument('--drop_prob', default=0,
                    help='model name')
parser.add_argument("--growth", type=int, default=32,
                    help="epoch with scale (1,1,1)")
parser.add_argument('--model_type', default='3d',
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


args = list(parser.parse_known_args())[0]
args.preload = True
args.debug = False
args.dir = "/storage"
args.sort = True
args.cuda = True
args.scale = (1,1,1)
args.offset = 3
args.stable_epoch = 1
args.tv_en = False
