import os
import time
import torch
import datetime
import math
import numpy as np
from tqdm import tnrange

import torch.nn as nn
from torchvision.utils import save_image
from torchsummary import summary
from tensorboardX import SummaryWriter
from sklearn.metrics import pairwise_distances as pdist

from utils.utils import *
from networks import CNN_network, SN_network, Attn_SN_network, ResNet_network, ResNet_SN_network, BN_SN_network


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader
        
        # Model hyper-parameters
        self.imsize = config.imsize
        self.z_dim = config.z_dim
        self.channel = config.channel
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_num = config.d_num
        self.topo_style = config.topo_style
        self.parallel = config.parallel
        self.network = config.network
        
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.use_tensorboard = config.use_tensorboard
        
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.loss_save_path = config.loss_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.printnet = config.printnet

        self.sigma = 1.0

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        # use tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter('./output/logs/' + self.version)

    def train(self):
        
        names = self.__dict__
        
        # Data iterator
        data_iter = iter(self.data_loader)

        # Fixed 64 input for generating samples
        fixed_z = torch.randn(64, self.z_dim).cuda()

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

            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)
            
            real_images = real_images.cuda()
            z = torch.randn(real_images.size(0), self.z_dim).cuda()
            

            # ================== Train D ================== #
            d_loss_real = 0
            d_loss_fake = 0
            
            D_real_vals = torch.empty(self.d_num, self.batch_size).cuda()

            for i in range(self.d_num):
                D_real_vals[i] = names['D_' + str(i)](real_images)
            Weight = self.ComputeWeight(D_real_vals, step)
            with torch.no_grad():
                fake_images = self.G(z)
            
            for i in range(self.d_num):
#                 di_loss_real = (Weight[i] * torch.log(D_real_vals[i])).mean()
                di_loss_real = (Weight[i] * torch.nn.ReLU()(1.0 - D_real_vals[i])).mean()
                d_loss_real += di_loss_real        
                
#                 di_loss_fake = - torch.log(names['D_' + str(i)](fake_images)).mean()
                di_loss_fake = torch.nn.ReLU()(1.0 + names['D_' + str(i)](fake_images)).mean()
                d_loss_fake += di_loss_fake
            
            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake #/ self.batch_size
            D_losses.append(d_loss.item())
            if self.use_tensorboard:
                self.writer.add_scalar('d_loss', d_loss.item(), step)
            self.reset_grad()
            d_loss.backward()
            for i in range(self.d_num):
                names['d_optimizer_' + str(i)].step()
            
            # ================== Train G  ================== #
            g_loss = 0
            
            fake_images = self.G(z)
            for i in range(self.d_num):
#                 gi_loss = torch.log(names['D_' + str(i)](fake_images)).mean()
                gi_loss = - names['D_' + str(i)](fake_images).mean()
                g_loss += gi_loss
            
            # Backward + Optimize
            self.reset_grad()
            G_losses.append(g_loss.item())
            if self.use_tensorboard:
                self.writer.add_scalar('g_loss', g_loss.item(), step)
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
                fake_images = self.G(fixed_z)
                save_image(denorm(fake_images.detach()), os.path.join(self.sample_path, 'G_{}.png'.format(step + 1)))
                save_loss(G_losses, self.loss_save_path, "G")
                save_loss(D_losses, self.loss_save_path, "D")
            
            # save models & loss
            if (step+1) % self.model_save_step==0:
                for i in range(self.d_num):
                    torch.save(names['D_' + str(i)].state_dict(), os.path.join(self.model_save_path, 'D{}_step{}.pth'.format(i, (step+1))))
                torch.save(self.G.state_dict(), os.path.join(self.model_save_path, 'G_step{}.pth'.format(step+1)))
                np.save(os.path.join(self.loss_save_path, 'G_losses'), G_losses)
                np.save(os.path.join(self.loss_save_path, 'D_losses'), D_losses)
        if self.use_tensorboard:
            self.writer.close()

    def build_model(self):
        names = self.__dict__
        
        self.initDist()
        print("Initialization distances with {} topo style successfully.".format(self.topo_style))
        
        if self.network == 'CNN_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = CNN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = CNN_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()
        elif self.network == 'SN_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = SN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = SN_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()
        elif self.network == 'Attn_SN_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = Attn_SN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = Attn_SN_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()   
        elif self.network == 'BN_SN_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = BN_SN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = BN_SN_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()        
        elif self.network == 'ResNet_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = ResNet_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = ResNet_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()        
        elif self.network == 'ResNet_SN_network':
            for i in range(self.d_num):
                names['D_' + str(i)] = ResNet_SN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = ResNet_SN_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()   
        elif self.network == 'mix':
            for i in range(self.d_num):
                names['D_' + str(i)] = CNN_network.Discriminator(self.imsize, self.channel, self.d_conv_dim).cuda()
            self.G = ResNet_network.Generator(self.imsize, self.z_dim, self.channel, self.g_conv_dim).cuda()        
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
        print("Initialization optimizers successfully.")
            
        if self.printnet:
            print("\n=============================\nG summary:")
            summary(self.G, (self.z_dim, 1, 1))                        
            print("D summary:")
            summary(self.D_0, (self.channel, self.imsize, self.imsize))
            # self.G.cuda()
            # self.D_0.cuda()
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
                    
        self.fixeddist = torch.from_numpy(fixeddist).cuda()
        
    def ComputeWeight(self, D_vals, step):
        max = torch.argmax(D_vals, dim=0)
        dist = torch.empty(self.d_num, self.batch_size).cuda()
        for i in range(self.batch_size):
            dist[:,i] = self.fixeddist[:,max[i]]
        self.sigma = self.sigma / (1 + step/(self.total_step**2))
        weight = torch.exp(-dist/(2*self.sigma**2))
        # weight_norm = weight / weight.sum(1, keepdim=True)
        return weight