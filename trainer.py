# +
import os
import utility
import torch
import numpy as np
from decimal import Decimal
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs


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
        self.patience_count = 0
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

        # self.optimizer.zero_grad()
                
        
        for batch, (lr,hr,scale,rel_coor) in enumerate(self.loader.training_data):

            self.optimizer.zero_grad()
            pbar.update(1)
            lr_tensor = lr.squeeze().to('cuda').float()  # ranges from [0, 1]
            hr_tensor = hr.squeeze().to('cuda').float()  # ranges from [0, 1]
            scale = np.asarray(scale[0,:])

            if(len(rel_coor) == 2):
                tv_tensor = rel_coor[1].squeeze().to('cuda').float()  
                rel_coor = rel_coor[0].squeeze().to('cuda').float()
            else:
                rel_coor = rel_coor.squeeze().to('cuda').float()
            
            if(len(lr_tensor.shape) == 5):
                lr = torch.permute(lr_tensor, (0,4,1,2,3))
            else:
                lr = torch.permute(lr_tensor, (0,3,1,2))


            # inference

            if(self.args.tv):
                pred,pred_tv = self.model.forward(lr,scale,rel_coor)
            else:
                pred = self.model.forward(lr,scale,rel_coor)
            if(len(lr_tensor.shape) == 5):
                # print(pred.shape,lr.shape)
                pred_tensor = torch.permute(pred, (0,2,3,4,1)).float()
                if(self.args.tv):
                    pred_tv_tensor = torch.permute(pred_tv, (0,2,3,4,1)).float()
            else:
                pred_tensor = torch.permute(pred, (0,2,3,1)).float()
                if(self.args.tv):
                    pred_tv_tensor = torch.permute(pred_tv, (0,2,3,1)).float()
            
            
            # loss function
            if(self.args.tv):
                loss = self.loss(pred_tensor,hr_tensor,pred_tv_tensor,tv_tensor)
            else:
                loss = self.loss(pred_tensor,hr_tensor)
            
            
            # backward
            if loss.item() < (self.args.skip_threshold * self.error_last):
                loss.backward()
                self.optimizer.step()
                if(self.logger != None):
                    self.logger.add_scalar("Loss",loss.item(),self.cnt)
                    self.cnt+=1
            
            
            pbar.set_description(f"{self.loss.display_loss(batch)}")
            pbar.set_postfix({"scale":scale,"blk_size":list(lr_tensor.shape),"non_zero":len(pred_tensor[pred_tensor>0])})
                        
            
        
            if(self.logger != None and np.random.randint(4) == 1):
            ## plotting                
                utility.plot_train_pred(lr_tensor,hr_tensor,pred_tensor,self.logger,self.iter,self.curr_epoch)
                self.iter +=1
            
        
        self.loss.end_log(len(self.loader.training_data))
        self.error_last = self.loss.log[-1, -1]
        
        
        # self.scheduler.step()
        self.loss.step()
        pbar.close()
    
        self.curr_epoch+=1

    def test(self):



        self.model.eval()

        eval_psnr_avg = []
        eval_hfen_avg = []
        
        pbar = tqdm(total = len(self.loader.testing_data))
        
        self.iter = 0
        # samples = random.sample(range(len(self.loader.testing_data)), num_samples)
        
        for _, (lr, hr,scale,rel_coor,out) in enumerate(self.loader.testing_data, 1):
            # print(lr.shape,hr.shape)
            pbar.update(1)
            lr_tensor = lr.squeeze().to('cuda').float()  # ranges from [0, 1]
            hr_tensor = hr.squeeze().to('cuda').float()  # ranges from [0, 1]
            out_tensor = out.squeeze().to('cuda').float()
            rel_coor = rel_coor.squeeze().to('cuda').float()
            scale = np.asarray(scale[0,:])

            # print(lr_tensor.shape,out_tensor.shape,hr_tensor.shape)
            
            
            if(len(lr_tensor.shape) == 5):
                lr = torch.permute(lr_tensor, (0,4,1,2,3))
            else:
                lr = torch.permute(lr_tensor, (0,3,1,2))
                
            
            with torch.no_grad():
                
                if(self.args.tv):
                    pred,_ = self.model.forward(lr,scale,rel_coor)
                else:
                    pred = self.model.forward(lr,scale,rel_coor)
                # pred = self.model.forward(lr,scale)
                # pred = torch.nn.functional.interpolate(out,hr_tensor.shape[1:-1])
                if(len(lr_tensor.shape) == 5):
                    pred_tensor = torch.permute(pred, (0,2,3,4,1)).float()
                else:
                    pred_tensor = torch.permute(pred, (0,2,3,1)).float()
            
            # print()
            if(self.logger != None and np.random.randint(4) == 1):
                # print("fig added")
                psnr, hfen = utility.compute_scores(hr_tensor,pred_tensor,out_tensor,scale,self.logger,self.iter,mask = True,epoch = self.curr_epoch)
                self.iter +=1
            else:
                psnr, hfen = utility.compute_scores(hr_tensor,pred_tensor,out_tensor,scale,mask = True)
            
            
            eval_hfen_avg.append(hfen)
            eval_psnr_avg.append(psnr)
            pbar.set_postfix({"scale":scale,"blk_size":list(lr_tensor.shape),"hfen":hfen,"psnr":psnr,"non_zero":len(pred[pred>0])})
            torch.cuda.empty_cache()
            lr_tensor = None
        
        # print(len(eval_hfen_avg))
        eval_hfen_avg = np.mean(eval_hfen_avg)
        eval_psnr_avg = np.mean(eval_psnr_avg)
        
        if(self.logger != None):
            self.logger.add_histogram("HFEN_hist",np.asarray(eval_hfen_avg),self.test_cnt)
            self.logger.add_histogram("PSNR_hist",np.asarray(eval_psnr_avg),self.test_cnt)
            
        
            self.logger.add_scalar("HFEN_avg",eval_hfen_avg,self.test_cnt)
            self.logger.add_scalar("PSNR_avg",eval_psnr_avg,self.test_cnt)
            self.test_cnt+=1
        

        if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
            
            self.psnr_max = eval_psnr_avg
            
            torch.save(
                self.model.state_dict(),
                os.path.join(self.ckp.dir ,"model", 'model_{}_{:.4f}_{:.4f}.pt'.format(self.curr_epoch,eval_hfen_avg,eval_psnr_avg))
            )
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(self.ckp.dir ,"optimizer",'optimizer_{}.pt'.format(self.curr_epoch))
            )
        
        
        
        if(self.patience_count > self.args.patience):
            self.patience_count = 0 
            t = self.loader.rebuild()
            self.logger.add_scalar("sca",t['sca'],self.test_cnt)
            self.logger.add_scalar("asy",t['asy'],self.test_cnt)
            self.logger.add_scalar("var",t['var'],self.test_cnt)
        
        self.patience_count+=1
                

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

