from option import args
import torch
import utility
import data
import utils
import model
import loss
from trainer import Trainer
import os
from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


ids = utils.get_ids()
ids.sort()
args.no_vols = 40
args.growth = 16
total_vols = args.no_vols
ids = ids[:total_vols]

if (args.run_name == '..'):
    args.run_name = f"{args.epochs}_epoch,{args.no_vols}_vols,blk_{args.block_size},loss_{args.loss},growth_{args.growth},new_hist_runs"
print(args.run_name)

# print(args.test_block_size)
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args,ids= ids)  
        logger = SummaryWriter('runs/'+ args.run_name)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint,logger)
        while not t.terminate():
            t.train()
            t.test()
            # break

        # checkpoint.done()

