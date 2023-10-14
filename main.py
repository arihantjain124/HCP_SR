from option import args
import torch
import utility
import data
import utils
import model
import loss
from trainer import Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ids = utils.get_ids()
ids.sort()
total_vols = 20
ids = ids[:total_vols]

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args,ids= ids)                ## data loader
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        # checkpoint.done()

