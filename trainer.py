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

        for batch, (lr, hr) in enumerate(self.loader.training_data):
            lr = lr.to('cuda').float()  # ranges from [0, 1]
            hr = hr.to('cuda').float()  # ranges from [0, 1]
            t3_vol = lr.shape
            t7_vol = hr.shape
            sca = [t7_vol[i]/t3_vol[i] for i in range(1,4)]
            timer_data.hold()
            self.optimizer.zero_grad()
            lr = torch.permute(lr, (0,4,1,2,3))
            # inference
            pred = self.model.forward(lr,sca)
            pred = torch.permute(pred, (0,2,3,4,1)).float()
            # loss function
            loss = self.loss(pred,hr)

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

    # def test(self):
    #     self.model.eval()

    #     with torch.no_grad():
    #         if self.args.test_only:
    #             scale_list = range(len(self.args.scale))
    #             logger = print
    #         else:
    #             scale_list = [9,19,29]
    #             logger = self.ckp.write_log

    #         eval_psnr_avg = []
    #         for idx_scale in scale_list:
    #             self.loader_test.dataset.set_scale(idx_scale)
    #             scale = self.args.scale[idx_scale]
    #             scale2 = self.args.scale2[idx_scale]

    #             eval_psnr = 0
    #             eval_ssim = 0
    #             for idx_img, (lr, hr, filename, _) in tqdm(enumerate(self.loader_test),total=len(self.loader_test)):
    #                 filename = filename[0]
    #                 # prepare LR & HR images
    #                 no_eval = (hr.nelement() == 1)
    #                 if not no_eval:
    #                     if isinstance(lr,list):
    #                         lr, ref_hr, ref_lr, hr = self.prepare(lr[0], lr[1], lr[2], hr)
    #                     else:
    #                         lr, hr = self.prepare(lr, hr)
    #                         ref_hr = None
    #                         ref_lr = None
    #                 else:
    #                     if isinstance(lr,list):
    #                         lr, ref_hr, ref_lr = self.prepare(lr[0], lr[1], lr[2])
    #                     else:
    #                         lr, = self.prepare(lr)
    #                         ref_hr = None
    #                         ref_lr = None
    #                 lr, hr, ref_hr, ref_lr = self.crop_border(lr, hr, ref_hr, ref_lr, scale, scale2)
    #                 # inference
    #                 self.model.get_model().set_scale(scale, scale2)
    #                 if ref_hr is None:
    #                     sr = self.model(lr)
    #                 else:
    #                     sr = self.model((lr, ref_hr, ref_lr, self.args.ref_type_test))
    #                 if isinstance(sr,tuple):
    #                     sr,Refsr = sr                    

    #                 if not no_eval:
    #                     psnr, ssim, mse = utility.calc_psnr(
    #                         lr, sr,  hr, img_name=filename, scale=[scale, scale2], 
    #                         save = self.args.save_results, savefile = self.args.savefigfilename,ref = ref_hr
    #                     )
    #                     eval_psnr += psnr
    #                     eval_ssim += ssim 


    #             if scale == scale2:
    #                 logger('[{} x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
    #                     self.args.data_test,
    #                     scale,
    #                     eval_psnr / len(self.loader_test),
    #                     eval_ssim / len(self.loader_test),
    #                 ))
    #             else:
    #                 logger('[{} x{}/x{}]\tPSNR: {:.4f} SSIM: {:.4f}'.format(
    #                     self.args.data_test,
    #                     scale,
    #                     scale2,
    #                     eval_psnr / len(self.loader_test),
    #                     eval_ssim / len(self.loader_test),
    #                 ))
    #             eval_psnr_avg.append(eval_psnr / len(self.loader_test))
    #         eval_psnr_avg = np.mean(eval_psnr_avg)
    #     if not self.args.test_only: #training mode and save the best model
    #         if self.psnr_max is None or self.psnr_max < eval_psnr_avg:
    #             self.psnr_max = eval_psnr_avg
    #             torch.save(
    #                 self.model.state_dict(),
    #                 os.path.join(self.ckp.dir, 'model', 'model_best.pt')
    #             )
    #             logger('save ckpt PSNR:{:.4f}'.format(eval_psnr_avg))

    def terminate(self):
        epoch = self.scheduler.last_epoch + 1
        return epoch >= self.args.epochs
