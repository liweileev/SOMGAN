import os
import time
import torch
import datetime
import math
import numpy as np
from tqdm import tnrange

import torch.nn as nn
from torchsummary import summary
from sklearn.metrics import pairwise_distances as pdist

from utils.utils import *
import matplotlib.pyplot as plt
from networks.toy2D_network import Generator, Discriminator

class Trainer(object):
    def __init__(self, config):
        
        # Model hyper-parameters
        self.z_dim = config.z_dim
        self.h_dim = config.h_dim
        self.d_num = config.d_num
        self.mix_coeffs = config.mix_coeffs
        self.mean = config.mean
        self.cov = config.cov
        self.num_samples = config.num_samples
        self.topo_style = config.topo_style
        self.parallel = config.parallel
        
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.loss_save_path = config.loss_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.printnet = config.printnet

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):
        
        names = self.__dict__

        # Fixed input for debugging
        fixed_z = torch.randn(self.batch_size, self.z_dim).cuda()

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
            
        # Storage losses for plot
        if self.pretrained_model:
            G_losses = np.load(os.path.join(self.loss_save_path, 'G_losses.npy')).tolist()
            D_losses = np.load(os.path.join(self.loss_save_path, 'D_losses.npy')).tolist()
        else:
            G_losses = []
            D_losses = []

        # Start time
        start_time = time.time()
        
        for step in tnrange(start, self.total_step):
            
            for i in range(self.d_num):
                names['D_' + str(i)].train()
            self.G.train()

            real_data = gmm_sample(self.batch_size, self.mix_coeffs, self.mean, self.cov)
            
            real_data = real_data.cuda()
            z = torch.randn(self.batch_size, self.z_dim).cuda()            

            # ================== Train D ================== #
            d_loss_real = 0
            d_loss_fake = 0
            
            D_real_vals = torch.empty(self.d_num, self.batch_size).cuda()

            for i in range(self.d_num):
                D_real_vals[i] = names['D_' + str(i)](real_data)
            Weight = self.ComputeWeight(D_real_vals, step)
            with torch.no_grad():
                fake_data = self.G(z)
            
            for i in range(self.d_num):
                # Hinge Loss：
                di_loss_real = (Weight[i] * torch.nn.ReLU()(1.0 - D_real_vals[i])).mean()
                d_loss_real += di_loss_real        
                
                di_loss_fake = torch.nn.ReLU()(1.0 + names['D_' + str(i)](fake_data)).mean()
                d_loss_fake += di_loss_fake
            
            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake / self.batch_size
            D_losses.append(d_loss.item())
            self.reset_grad()
            d_loss.backward()
            for i in range(self.d_num):
                names['d_optimizer_' + str(i)].step()
            
            # ================== Train G  ================== #
            g_loss = 0
            
            fake_data = self.G(z)
            for i in range(self.d_num):
                gi_loss = - names['D_' + str(i)](fake_data).mean()
                g_loss += gi_loss
            
            # Backward + Optimize
            self.reset_grad()
            G_losses.append(g_loss.item())
            g_loss.backward()
            self.g_optimizer.step()
            
            # ================== log and save  ================== #
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(elapsed, step + 1, self.total_step, d_loss.item(), g_loss.item()))

            # Sample images
            if step==0 or (step + 1) % self.sample_step == 0:
                fake_data = self.G(fixed_z)
                self.save_image(step)
                save_loss(G_losses, self.loss_save_path, "G")
                save_loss(D_losses, self.loss_save_path, "D")
            
            # save models
            if step==0 or (step + 1) % self.model_save_step == 0:
                for i in range(self.d_num):
                    torch.save(names['D_' + str(i)].state_dict(), os.path.join(self.model_save_path, 'D{}_step{}.pth'.format(i, (step+1))))
                torch.save(self.G.state_dict(), os.path.join(self.model_save_path, 'G_step{}.pth'.format(step+1)))
                np.save(os.path.join(self.loss_save_path, 'G_losses'), G_losses)
                np.save(os.path.join(self.loss_save_path, 'D_losses'), D_losses)

    def build_model(self):
        names = self.__dict__
        
        self.initDist()
        print("Initialization distances with {} topo style successfully.".format(self.topo_style))
        
        for i in range(self.d_num):
            names['D_' + str(i)] = Discriminator(self.h_dim).cuda()
        self.G = Generator(self.z_dim, self.h_dim).cuda()
        print("Initialization parameters of Generator & Discriminator successfully.")
            
        if self.parallel:
            for i in range(self.d_num):
                names['D_' + str(i)] = nn.DataParallel(names['D_' + str(i)])
            self.G = nn.DataParallel(self.G)
            print("Parallel computing started.")

       # Loss and optimizer
        for i in range(self.d_num):
            names['d_optimizer_' + str(i)] = torch.optim.Adam(filter(lambda p: p.requires_grad, names['D_' + str(i)].parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        print("Initialization optimizers of Generator & Discriminator successfully.")
            
        if self.printnet:
            print("\n=============================\nG summary:")
            summary(self.G.cpu(), (self.z_dim, 1))
            print("D summary:")
            summary(self.D_0.cpu(), (2, 1))
            self.G.cuda()
            self.D_0.cuda()
            print("\n=============================\n")

    def load_pretrained_model(self):
        names = self.__dict__
        for i in range(self.d_num):
            names['D_' + str(i)].load_state_dict(torch.load(os.path.join(self.model_save_path, 'D{}_step{}.pth'.format(i, self.pretrained_model))))
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, 'G_step{}.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        names = self.__dict__
        for i in range(self.d_num):
            names['d_optimizer_' + str(i)].zero_grad()
        self.g_optimizer.zero_grad()
    
    def initDist(self):
        # grid
        if self.topo_style == 'grid':
            coord = np.zeros((self.d_num, 2))
            ind = 0
            for i in range(int(math.sqrt(self.d_num))):
                for j in range(int(math.sqrt(self.d_num))): 
                    coord[ind,:] = [i, j]
                    ind += 1
            fixeddist = pdist(coord, coord)
            
        # linear
        elif self.topo_style == 'linear':
            fixeddist = np.zeros((self.d_num, self.d_num))
            for i in range(self.d_num):
                for j in range(self.d_num):
                    fixeddist[i][j] = abs(i-j)
        
        # circular
        elif self.topo_style == 'circular':
            #  Geodesic Distance
            fixeddist = np.zeros((self.d_num, self.d_num))
            for i in range(self.d_num):
                for j in range(self.d_num):
                    fixeddist[i][j] = 2*math.pi/self.d_num * min(abs(i-j),(self.d_num-abs(i-j)))
                    
#             # Euclidean distance
#             coord = np.zeros((self.d_num, 2))
#             coord[:,0] = [math.cos((i+1)*2*math.pi/self.d_num) for i in range(self.d_num)]
#             coord[:,1] = [math.sin((i+1)*2*math.pi/self.d_num) for i in range(self.d_num)]
#             fixeddist = pdist(coord, coord)
            
        self.fixeddist = torch.from_numpy(fixeddist).cuda()
        
    def ComputeWeight(self, D_vals, step):
        max = torch.argmax(D_vals, dim=0)
        dist = torch.empty(self.d_num, self.batch_size).cuda()
        for i in range(self.batch_size):
            dist[:,i] = self.fixeddist[:,max[i]]
        weight = torch.exp(-dist)
        weight_norm = weight / weight.sum(1, keepdim=True)
        return weight_norm

    def save_image(self, step):
        names = self.__dict__
        real_data_num_samples = gmm_sample(self.num_samples, self.mix_coeffs, self.mean, self.cov)
        z = torch.randn(self.num_samples, self.z_dim).cuda()  
        with torch.no_grad():
                fake_data_num_samples = self.G(z)    
        
        # divide fake data by discriminators
        D_result_fake = torch.empty(self.d_num, self.num_samples).cuda()
        for i in range(self.d_num):
            D_result_fake[i] = names['D_' + str(i)](fake_data_num_samples)
        max_label = torch.argmax(D_result_fake, dim=0)
        fake_data_num_samples = fake_data_num_samples.cpu()
        
        # plot & save
        plt.figure()
        plt.scatter(real_data_num_samples[:, 0], real_data_num_samples[:, 1], marker='+', c='r', label='real data')
#         plt.scatter(fake_data_num_samples[:, 0], fake_data_num_samples[:, 1], marker='o', c='b', label='generated data')
        for i in range(self.d_num):
            plt.scatter(fake_data_num_samples[max_label==i, 0], fake_data_num_samples[max_label==i, 1], marker='o', label='generated data with D_{}'.format(i))
        plt.legend(loc=[1.05, 0])
        plt.savefig(os.path.join(self.sample_path, '{}.png'.format(step + 1)), bbox_inches='tight')
        plt.close()