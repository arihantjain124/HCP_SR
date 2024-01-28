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
        self.var_blk_size = args.var_blk_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.start_stable = args.start_stable
        print(self.batch_size)
        self.desc_blk_size = [(32,32,8),(16,16,4)]
        if(self.var_blk_size):
            self.ord = 0
            self.last_5 = []

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.psnr_max = None

    def train(self):

        self.loss.start_log()
        self.model.train()
        
        lr = self.scheduler.get_last_lr()[0]
        if(self.logger != None):
            self.logger.add_scalar("LR",lr,self.curr_epoch)
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(self.curr_epoch, Decimal(lr)))
        
        pbar = tqdm(total = len(self.loader.training_data))
        
        self.iter = 0
        self.optimizer.zero_grad()
        
        lr_tensor = None
        hr_tensor = None
        curr_scale = None
        for batch, (lr, hr,scale) in enumerate(self.loader.training_data):
            pbar.update(1)
            ltensor = lr.to('cuda').float()  # ranges from [0, 1]
            htensor = hr.to('cuda').float()  # ranges from [0, 1]
            
            if(lr_tensor is None):
                lr_tensor = ltensor.to('cuda')
                hr_tensor = htensor.to('cuda')
                curr_scale = scale
                continue
            elif(lr_tensor.shape[1:] == ltensor.shape[1:] and lr_tensor.shape[0]<self.batch_size and (curr_scale == scale).all()):
                lr_tensor = torch.cat((lr_tensor,ltensor)).to('cuda')
                hr_tensor = torch.cat((hr_tensor,htensor)).to('cuda')
                continue
            else:
                
                self.optimizer.zero_grad()
                
                lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
                
                # inference
                pred = self.model.forward(lr_tensor,curr_scale)
                
                pred = torch.permute(pred, (0,2,3,4,1)).float()

                # loss function
                loss = self.loss(pred,hr_tensor)
                
                
                # backward
                if loss.item() < self.args.skip_threshold * self.error_last:
                    loss.backward()
                    
                    if(self.logger != None):
                        self.logger.add_scalar("Loss",loss.item(),self.cnt)
                        self.cnt+=1
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        batch + 1, loss.item()
                    ))
                
                pbar.set_description(f"{self.loss.display_loss(batch)}")
                pbar.set_postfix({"scale":scale,"blk_size":list(lr_tensor.shape)})
                            
                self.optimizer.step()
            
            
                if (np.random.randint(2) == 1):
                ## plotting                
                    lr_tensor = torch.permute(lr_tensor, (0,2,3,4,1)).float()
                    utility.plot_train_pred(lr_tensor,hr_tensor,pred,self.logger,self.iter,self.curr_epoch)
                    self.iter +=1
        
            lr_tensor = None
            
        self.loss.end_log(len(self.loader.training_data))
        self.error_last = self.loss.log[-1, -1]
        
        self.scheduler.step()
        # self.loss.step()
        pbar.close()
    
    

    # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if (self.curr_epoch+1) < self.args.stable_epoch:
            self.loader.rebuild(type = "train",train_var = False)
            
        elif (((self.curr_epoch+1)%self.args.offset == 0)):
            self.loader.rebuild(type = "train",train_var = self.var_blk_size)
            
        self.curr_epoch+=1

    def test(self):
        self.model.eval()
        eval_psnr_avg = []
        eval_hfen_avg = []
        
        pbar = tqdm(total = len(self.loader.testing_data))
        
        self.iter = 0
        lr_tensor = None
        
        for _, (lr, hr,out,scale) in enumerate(self.loader.testing_data, 1):
            # print(lr_tensor.shape,hr_tensor.shape)
            pbar.update(1)
            ltensor = lr.to('cuda').float()  # ranges from [0, 1]
            htensor = hr.to('cuda').float()  # ranges from [0, 1]
            otensor = out.to('cuda').float()
            if(lr_tensor is None):
                lr_tensor = ltensor.to('cuda')
                hr_tensor = htensor.to('cuda')
                out_tensor = otensor.to('cuda')
                curr_scale = scale
                continue
            elif(lr_tensor.shape[1:] == ltensor.shape[1:] and lr_tensor.shape[0]<self.test_batch_size and (curr_scale == scale).all()):
                lr_tensor = torch.cat((lr_tensor,ltensor)).to('cuda')
                hr_tensor = torch.cat((hr_tensor,htensor)).to('cuda')
                out_tensor = torch.cat((out_tensor,otensor)).to('cuda')
                continue
            elif(lr_tensor.shape[0]>1):
                # print(lr_tensor.shape)
                lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
                        
                with torch.no_grad():
                    pred = self.model.forward(lr_tensor,curr_scale)
                    pred = torch.permute(pred, (0,2,3,4,1)).float()
                
                
                if(self.logger != None and np.random.randint(2) == 1):
                    # print("fig added")
                    psnr, hfen = utility.compute_scores(hr_tensor,pred,out_tensor,scale,self.logger,self.iter,mask = True,epoch = self.curr_epoch)
                    self.iter +=1
                else:
                    psnr, hfen = utility.compute_scores(hr_tensor,pred,out_tensor,scale,mask = True)
                
                
                eval_hfen_avg.append(hfen)
                eval_psnr_avg.append(psnr)
                pbar.set_postfix({"scale":scale,"blk_size":list(lr_tensor.shape[2:]),"hfen":hfen})
                torch.cuda.empty_cache()
                lr_tensor = None
        
        # print(len(eval_hfen_avg))
        
        if(self.logger != None):
            self.logger.add_histogram("HFEN_hist",np.asarray(eval_hfen_avg),self.test_cnt)
            self.logger.add_histogram("PSNR_hist",np.asarray(eval_psnr_avg),self.test_cnt)
            
        eval_hfen_avg = np.mean(eval_hfen_avg)
        eval_psnr_avg = np.mean(eval_psnr_avg)
        
        if(self.logger != None):
            self.logger.add_scalar("HFEN_avg",eval_hfen_avg,self.test_cnt)
            self.logger.add_scalar("PSNR_avg",eval_psnr_avg,self.test_cnt)
            self.test_cnt+=1
        
        
        if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
            self.psnr_max = eval_psnr_avg
            torch.save(
                self.model.state_dict(),
                os.path.join(self.ckp.dir, 'model', f"model_best_{self.args.run_name}.pt")
            )
                

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
