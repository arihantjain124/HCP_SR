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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ids = utils.get_ids()
ids.sort()
total_vols = args.no_vols
ids = ids[:total_vols]
# print(args.test_block_size)
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args,ids= ids)  
        logger = SummaryWriter('runs/'+ args.run_name)
        print(args.run_name)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint,logger)
        while not t.terminate():
            t.train()
            t.test()
            # break

        # checkpoint.done()

