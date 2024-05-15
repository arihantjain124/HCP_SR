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




args.run_name = "incremental range increase"
if(args.model_type == '2d'):
    args.block_size = (32,32,1)

if (args.run_name == '..'):
    args.run_name = f"{args.model},attn_{args.attention},{args.model_type},{args.encoder},{args.no_vols}_train,{args.test_vols}_test,growth{args.growth},loss_{args.loss},start_var_{args.start_var},batch_size{args.batch_size}"
else:
    args.run_name = f"{args.model},attn_{args.attention},{args.model_type},{args.encoder},{args.no_vols}_train,{args.test_vols}_test,growth{args.growth},loss_{args.loss},start_var_{args.start_var},batch_size{args.batch_size},{args.run_name}"
print(args.run_name)

# print(args.test_block_size)






from tqdm import tqdm

if __name__ == '__main__':
    curr_epoch = 0
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)       ## setting the log and the train information
    if checkpoint.ok:
        loader = data.Data(args,ids= ids)  
        logger = SummaryWriter('runs/'+ args.run_name)
        model = model.Model(args)
        loss_fn = loss.Loss(args, checkpoint)

        optimizer = utility.make_optimizer(args, model)
        scheduler = utility.make_scheduler(args, optimizer)
        
        cnt = 0

        while curr_epoch <= args.epochs:

            pbar = tqdm(total = len(loader.training_data))
            model.train()
            lr = scheduler.get_last_lr()[0]
            logger.add_scalar("LR",lr,curr_epoch)
            iter = 0
            for batch, (lr,hr,scale) in enumerate(loader.training_data):

                pbar.update(1)
            
                lr_tensor = lr.squeeze().to('cuda').float()  # ranges from [0, 1]
                hr_tensor = hr.squeeze().to('cuda').float()  # ranges from [0, 1]
                scale = np.asarray(scale[0,:])

                
                if(len(lr_tensor.shape) == 5):
                    lr = torch.permute(lr_tensor, (0,4,1,2,3))
                else:
                    lr = torch.permute(lr_tensor, (0,3,1,2))


                # inference
                pred = model.forward(lr,scale)

                pred_tensor = torch.permute(pred, (0,2,3,4,1)).float()
                

                loss_curr = loss_fn(pred_tensor,hr_tensor)
                loss_curr.backward()
                logger.add_scalar("Loss",loss_curr.item(),cnt)
                cnt+=1

                pbar.set_description(f"{loss_curr.item()}")
                pbar.set_postfix({"scale":scale,"blk_size":list(lr_tensor.shape),"non_zero":len(pred_tensor[pred_tensor>0])})
                            
                optimizer.step()

                # if(np.random.randint(4) == 1):
                #     ## plotting                
                #     utility.plot_train_pred(lr_tensor,hr_tensor,pred_tensor,logger,iter,curr_epoch)
                #     iter +=1

                scheduler.step()
            pbar.close()
             
            curr_epoch+=1
        # t = Trainer(args, loader, model, loss, checkpoint,logger)
        # while not t.terminate():
        #     # t.train()
        #     # break
        #     t.test()
        #     # break

        # checkpoint.done()

