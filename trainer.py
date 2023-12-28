import os
import math
import matplotlib
matplotlib.use('Agg')
import utility
import torch
import numpy as np
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import random

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp,logger = None):
        self.args = args
        self.logger = logger
        self.ckp = ckp
        self.loader = loader
        self.model = my_model
        self.loss = my_loss
        self.cnt = 0
        self.test_cnt = 0
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.curr_epoch = 0

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.psnr_max = None

    def train(self):
        print(self.curr_epoch)

        self.loss.start_log()
        self.model.train()
        
        # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if (self.curr_epoch+1) < self.args.offset and self.args.load == '.':
            print("stablizing")
            
        if (((self.curr_epoch+1)%self.args.offset == 0)):
            print("destablizing")
            self.loader.rebuild(blk_size = (16,16,4),type = "train",stable = False)
            
        # if self.curr_epoch > self.args.offset:
        #     self.scheduler = lrs.OneCycleLR(
        #     self.optimizer,
        #     max_lr = self.args.max_lr,
        #     steps_per_epoch  = len(self.loader.training_data),
        #     epochs = 1
        #     )

        # train on all scale factors for remaining epochs

        lr = self.scheduler.get_last_lr()[0]
        if(self.logger != None):
            self.logger.add_scalar("LR",lr,self.curr_epoch)
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(self.curr_epoch, Decimal(lr)))
        
        pbar = tqdm(total = len(self.loader.training_data))
        num_samples = len(self.loader.training_data) //4
        samples = random.sample(range(len(self.loader.training_data)), num_samples)
        for batch, (lr_tensor, hr_tensor,scale) in enumerate(self.loader.training_data):
            pbar.update(1)
            lr_tensor = lr_tensor.to('cuda').float()  # ranges from [0, 1]
            hr_tensor = hr_tensor.to('cuda').float()  # ranges from [0, 1]
            
            self.optimizer.zero_grad()
            lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
            # inference
            pred = self.model.forward(lr_tensor,scale)
            pred = torch.permute(pred, (0,2,3,4,1)).float()

            # loss function
            loss = self.loss(pred,hr_tensor)
            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                
                if(self.logger != None):
                    self.logger.add_scalar("Loss",loss.item(),self.cnt)
                self.cnt+=1
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))


            if (batch + 1) % self.args.print_every == 0:
                pbar.set_description(f"{self.loss.display_loss(batch)}")
                pbar.set_postfix({"scale":scale})
                
            if (batch in samples and self.curr_epoch >0):
            ## plotting
                lr_tensor = torch.permute(lr_tensor,  (0,2,3,4,1))
                utility.plot_train_pred(lr_tensor,hr_tensor,self.logger,batch)
                
        self.loss.end_log(len(self.loader.training_data))
        self.error_last = self.loss.log[-1, -1]
        self.curr_epoch+=1
        self.scheduler.step()
        # self.loss.step()
        pbar.close()


    def test(self):
        self.model.eval()
        eval_psnr_avg = []
        eval_ssim_avg = []
        pbar = tqdm(total = len(self.loader.testing_data))
        num_samples = 2
        samples = random.sample(range(len(self.loader.testing_data)), num_samples)
        for iteration, (lr_tensor, hr_tensor,pnts,scale) in enumerate(self.loader.testing_data, 1):
            # print(lr_tensor.shape,hr_tensor.shape)
            pbar.update(1)
            pbar.set_postfix({"scale":scale})
            lr_tensor = lr_tensor.to(self.device)
            hr_tensor = hr_tensor.to(self.device)
            lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
            # inference
            # print(lr_tensor.shape)
            with torch.no_grad():
                pred = self.model.forward(lr_tensor,scale)
            pred = torch.permute(pred, (0,2,3,4,1)).float()
            # epoch = self.scheduler.last_epoch - 1 
            if(iteration in samples and self.logger != None):
                # print("fig added")
                psnr, ssim = utility.compute_psnr_ssim(hr_tensor,pred,pnts,self.logger,iteration,mask = self.args.test_mask)
            else:
                psnr, ssim = utility.compute_psnr_ssim(hr_tensor,pred,pnts,mask = self.args.test_mask)
            # eval_psnr += psnr
            # eval_ssim += ssim 
            # psnr,ssim = 0,0
            eval_ssim_avg.append(ssim)
            eval_psnr_avg.append(psnr)
            torch.cuda.empty_cache()
        eval_ssim_avg = np.mean(eval_ssim_avg)
        eval_psnr_avg = np.mean(eval_psnr_avg)
        if(self.logger != None):
            self.logger.add_scalar("SSIM",eval_ssim_avg,self.test_cnt)
            self.logger.add_scalar("PSNR",eval_psnr_avg,self.test_cnt)
        self.test_cnt+=1
        if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
            self.psnr_max = eval_psnr_avg
            torch.save(
                self.model.state_dict(),
                os.path.join(self.ckp.dir, 'model', f"model_best_{self.args.run_name}.pt")
            )
            
        torch.cuda.empty_cache()
                

    def save_model(self,epoch):
        
        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_latest.pt')
        )
        if epoch % self.args.save_every == 0:
            torch.save(
                target.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
            self.ckp.write_log('save ckpt epoch{:.4f}'.format(epoch))

    def terminate(self):
        return self.curr_epoch >= self.args.epochs
