'''
Author: Liweileev
Date: 2022-01-06 11:58:00
LastEditors: Liweileev
LastEditTime: 2022-01-10 01:50:43
'''

import numpy as np
import torch
import math
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from sklearn.metrics import pairwise_distances as pdist

class SOMGANLoss():
    def __init__(self, device, kimg, num_D, topo, G_mapping, G_synthesis, Ds, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.kimg = kimg
        self.num_D = num_D
        self.topo = topo
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.Ds = Ds
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.initDist()
        self.sigma = 1.0
        print("Initialization distances with {} topo constraint successfully.".format(self.topo))

    def run_G(self, z, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

            
    def run_D(self, D, img, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(D, sync):
            logits = D(img)
        return logits

    def accumulate_gradients(self, phase, real_img, gen_z, cur_nimg, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Dsmain', 'Dsreg']
        do_Gmain = phase == 'Gmain'
        do_Dmain = phase == 'Dsmain'
        do_Gpl   = phase == 'Greg' and (self.pl_weight != 0)
        do_Dr1   = phase == 'Dsreg' and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                loss_Gmain = 0
                gen_img, _ = self.run_G(gen_z, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                for i in range(self.num_D):
                    gen_logits = self.run_D(self.Ds['D'+str(i)], gen_img, sync=False)
                    training_stats.report('Loss/scores/fake_{i}', gen_logits)
                    training_stats.report('Loss/signs/fake_{i}', gen_logits.sign())
                    loss_Gmain += torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _ = self.run_G(gen_z, sync=False)
                for i in range(self.num_D):
                    gen_logits = self.run_D(self.Ds['D'+str(i)], gen_img, sync=False)
                    training_stats.report('Loss/scores/fake_{i}', gen_logits)
                    training_stats.report('Loss/signs/fake_{i}', gen_logits.sign())
                    loss_Dgen += torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
                
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = torch.empty(self.num_D, real_img_tmp.shape[0], 1, device=self.device)

                for i in range(self.num_D):
                    real_logits[i] = self.run_D(self.Ds['D'+str(i)], real_img_tmp, sync=sync)
                    training_stats.report('Loss/scores/real_{i}', real_logits[i])
                    training_stats.report('Loss/signs/real_{i}', real_logits[i].sign())
                Weight = self.ComputeWeight(real_logits.squeeze(2), cur_nimg)

                loss_Dreal = 0
                for i in range(self.num_D):
                    loss_Dreal += (Weight[i] * torch.nn.functional.softplus(-real_logits[i])) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


    def initDist(self):
        # grid
        if self.topo == 'grid':
            coord = np.zeros((self.num_D, 2))
            ind = 0
            for i in range(int(math.sqrt(self.num_D))):
                for j in range(int(math.sqrt(self.num_D))): 
                    coord[ind,:] = [i, j]
                    ind += 1
            fixeddist = pdist(coord, coord)
            
        self.fixeddist = torch.from_numpy(fixeddist).to(self.device)
    
    def ComputeWeight(self, D_vals, cur_nimg):
        max = torch.argmax(D_vals, dim=0)
        dist = torch.empty(self.num_D, D_vals.shape[1], device=self.device)
        for i in range(D_vals.shape[1]):
            dist[:,i] = self.fixeddist[:,max[i]]
        self.sigma = self.sigma / (1 + cur_nimg/((self.kimg*1000)**2))
        weight = torch.exp(-dist/(2*self.sigma**2))
        return weight
#----------------------------------------------------------------------------