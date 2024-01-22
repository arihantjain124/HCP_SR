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
total_vols = args.no_vols+args.test_vols
temp = ids[:total_vols]
# temp.extend(ids[offset:args.test_vols+offset])
ids = temp
# print(ids)
if(args.var_blk_size):
    args.run_name = f"{args.epochs}_epoch,{args.no_vols}_vols,{args.test_vols}_test_vols,blk_{args.block_size},loss_{args.loss},growth_{args.growth},desc_{args.var_blk_size}"
elif (args.run_name == '..'):
    args.run_name = f"{args.epochs}_epoch,{args.no_vols}_vols,{args.test_vols}_test_vols,blk_{args.block_size},loss_{args.loss},growth_{args.growth}"
else:
    args.run_name = f"{args.epochs}_epoch,{args.no_vols}_vols,{args.test_vols}_test_vols,blk_{args.block_size},loss_{args.loss},growth_{args.growth},{args.run_name}"
print(args.run_name)

# print(args.test_block_size)
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args,ids= ids)  
        logger = SummaryWriter('runs/'+ args.run_name)
        model = model.Model(args)
        loss = loss.Loss(args, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint,logger)
        while not t.terminate():
            t.train()
            if(t.curr_epoch >2):
                t.test()
            # break

        # checkpoint.done()

