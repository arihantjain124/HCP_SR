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
import numpy as np

np.random.seed(args.seed)
ids = utils.get_ids()
# ids.sort()
total_vols = args.no_vols+args.test_vols
ids.sort()
ids = ids[:total_vols]
ids = np.random.choice(ids,total_vols,replace = False)
print(ids)
if(args.model_type == '2d'):
    args.block_size = (32,32,1)

if (args.run_name == '..'):
    args.run_name = f"{args.model},train_{args.no_vols},test_{args.test_vols},{args.RDNconfig},{args.attention},{args.encoder},growth{args.growth},loss_{args.loss},bs_{args.batch_size},tv{args.tv}"
else:
    args.run_name = f"{args.model},train_{args.no_vols},test_{args.test_vols},{args.RDNconfig},{args.attention},{args.encoder},growth{args.growth},loss_{args.loss},bs_{args.batch_size},tv{args.tv},{args.run_name}"
print(args.run_name)

# print(args.test_block_size)
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        model = model.Model(args)
        print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + "  Number of Parameters")
        loader = data.Data(args,ids= ids)  
        logger = SummaryWriter('runs/'+ args.run_name)
        loss = loss.Loss(args, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint,logger)
        while not t.terminate():
            t.train()
            t.test()
            # break

        # checkpoint.done()

