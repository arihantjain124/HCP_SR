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

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader = loader
        self.model = my_model
        self.loss = my_loss
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.psnr_max = None

    def train(self):
        self.loss.step()
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if epoch == 1 and self.args.load == '.':
            # adjust learning rate
            lr = 5e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # train on all scale factors for remaining epochs
        else:
            # adjust learning rate
            lr = self.args.lr * (2 ** -(epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        pbar = tqdm(total = len(self.loader.training_data))
        for batch, (lr_tensor, hr_tensor) in enumerate(self.loader.training_data):
            pbar.update(1)
            lr_tensor = lr_tensor.to('cuda').float()  # ranges from [0, 1]
            hr_tensor = hr_tensor.to('cuda').float()  # ranges from [0, 1]
            t3_vol = lr_tensor.shape
            t7_vol = hr_tensor.shape
            sca = [t7_vol[i]/t3_vol[i] for i in range(1,4)]
            timer_data.hold()
            self.optimizer.zero_grad()
            lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
            # inference
            pred = self.model.forward(lr_tensor,sca)
            pred = torch.permute(pred, (0,2,3,4,1)).float()
            # loss function
            loss = self.loss(pred,hr_tensor)

            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader.training_data),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader.training_data))
        self.error_last = self.loss.log[-1, -1]

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
        pbar.close()


    def test(self):
        self.model.eval()
        eval_psnr_avg = []
        eval_ssim_avg = []
        pbar = tqdm(total = len(self.loader.testing_data))
        eval_psnr = 0
        eval_ssim = 0
        for iteration, (lr_tensor, hr_tensor,pnts,mask) in enumerate(self.loader.testing_data, 1):
            # print(lr_tensor.shape,hr_tensor.shape)
            pbar.update(1)
            lr_tensor = lr_tensor.to(self.device)
            hr_tensor = hr_tensor.to(self.device)
            sca = self.loader.get_scale_test()
            lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
            # inference
            with torch.no_grad():
                pred = self.model.forward(lr_tensor,sca)
            pred = torch.permute(pred, (0,2,3,4,1)).float()


            psnr, ssim = utility.compute_psnr_ssim(hr_tensor,pred,pnts,mask)
            eval_psnr += psnr
            eval_ssim += ssim 

            eval_ssim_avg.append(eval_ssim)
            eval_psnr_avg.append(eval_psnr)
        eval_ssim_avg = np.mean(eval_ssim_avg)
        eval_psnr_avg = np.mean(eval_psnr_avg)
        if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
            self.psnr_max = eval_psnr_avg
            torch.save(
                self.model.state_dict(),
                os.path.join(self.ckp.dir, 'model', 'model_best.pt')
            )
                
    def terminate(self):
        epoch = self.scheduler.last_epoch + 1
        return epoch >= self.args.epochs
