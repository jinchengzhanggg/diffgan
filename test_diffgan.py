# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np

import os

import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
from datasets_prep.midi_util import decode_sample_for_midi, save_piano_roll_midi

#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, label, opt):
    x =x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0_cond = generator(x, t_time, latent_z, label, train=False)
            x_0_uncond = generator(x, t_time, latent_z, label, train=False, uncondition=True)
            x_0 = (1 + opt.w) * x_0_cond - opt.w * x_0_uncond
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

#%%
def sample_and_test(args):
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    netG = NCSNpp(args).to(device)
    print(sum([i.numel() for i in netG.parameters()]))
    
    ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id)
                      ,weights_only=False, map_location=device)    
    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    
    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
        
    iters_needed = 50000 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_all = torch.randn(args.gen_number, args.num_channels, args.height, args.width)
                results = []
                labels = []
                for ind,classify_img in enumerate(torch.split(x_all,args.gen_number//args.class_num)):
                    for k in torch.split(classify_img, args.batch_size):
                        x_t_1 = k.to(device)
                        label = torch.randint(ind, ind+1, size=(x_t_1.shape[0],), device=device)
                        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, label, args)
                        fake_sample = to_range_0_1(fake_sample)
                        results.append(fake_sample)
                        labels.append(label)
                label = torch.cat(labels)
                fake_sample = torch.cat(results)
                if args.dataset == "pet4":
                    num = 0
                    label = label.tolist()
                    if not os.path.exists("./data/Generated_image"):
                        os.mkdir("./data/Generated_image")
                    index_label = {}

                    for i in open("./label.txt", "r").readlines():
                        a = i.strip().split(":")
                        index_label[int(a[0])] = a[1]

                    for idx, img in enumerate(fake_sample):
                        label_str = index_label[label[idx]]
                        if not os.path.exists(os.path.join("./data/Generated_image", label_str)):
                            os.mkdir(os.path.join("./data/Generated_image", label_str))
                        torchvision.utils.save_image(img,
                                                     './data/Generated_image/{}/samples_{}.jpg'.format(label_str, num))
                        num += 1
                else:
                    for j, x in enumerate(fake_sample):
                        index = i * args.batch_size + j
                        torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                    print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        x_all = torch.randn(args.gen_number, args.num_channels,args.height, args.width)
        results = []
        labels = []
        for ind,classify_img in enumerate(torch.split(x_all,args.gen_number//args.class_num)):
                    for k in torch.split(classify_img, args.batch_size):
                        x_t_1 = k.to(device)
                        label = torch.randint(ind, ind+1, size=(x_t_1.shape[0],), device=device)
                        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, label, args)
                        # fake_sample = to_range_0_1(fake_sample)
                        sample = decode_sample_for_midi(fake_sample, scale_factor=1., threshold=-0.95)
                        arr = sample.cpu().numpy()
                        arr = arr.transpose(0, 3, 1, 2)
                        
                        arr[:, :, :21, :] = 0
                        arr[:, :, 108 + 1:, :] = 0

                        img2 = np.append(arr, np.zeros((arr.shape[0], 1, arr.shape[2],arr.shape[3])), axis=1)
                        fake_sample = torch.from_numpy(img2)
                        results.append(fake_sample)
                        labels.append(label)
        label = torch.cat(labels)
        fake_sample = torch.cat(results)
        
        save_dir = os.path.join("./data", 'sample_discrete_epoch')
        os.makedirs(save_dir, exist_ok=True)
        save_piano_roll_midi(fake_sample.cpu().numpy(), save_dir, 8)
            
        if args.dataset == "pet4":
            num = 0
            label = label.tolist()
            if not os.path.exists("./data/Generated_image"):
                os.mkdir("./data/Generated_image")
            if not os.path.exists("./data/Generated_npy"):
                os.mkdir("./data/Generated_npy")
            index_label = {}

            for i in open("./label.txt","r").readlines():
                a = i.strip().split(":")
                index_label[int(a[0])] = a[1]
            for i in index_label.values():
                if not os.path.exists(os.path.join("./data/Generated_image",i)):
                    os.mkdir(os.path.join("./data/Generated_image",i))
                if not os.path.exists(os.path.join("./data/Generated_npy",i)):
                    os.mkdir(os.path.join("./data/Generated_npy",i))
            for idx,img in enumerate(fake_sample):
                label_str = index_label[label[idx]]
                np.save('./data/Generated_npy/{}/samples_{}.npy'.format(label_str,num), img[:2].numpy())
                torchvision.utils.save_image(img/127.0, './data/Generated_image/{}/samples_{}.jpg'.format(label_str,num))
                num+=1
        else:
            torchvision.utils.save_image(fake_sample, './samples_{}.jpg'.format(args.dataset))
            np.savetxt('./sample_{}.txt'.format(args.dataset), label.cpu())

    
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser('ddgan parameters')

    parser.add_argument("--class_num", type=int, default=4,
                        help="select a number of class")
    parser.add_argument("--w", type=float, default=3.0,
                        help='hyperparameters for classifier-free guidance strength')
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="threshold for classifier-free guidance")

    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(32,16,8,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=128,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8, help='sample generating batch size')
    parser.add_argument('--gen_number', type=int, default=100, help='sample generating number')

    parser.add_argument('--width', type=int, default=256, help=' generating width')
    parser.add_argument('--height', type=int, default=128, help=' generating height')




   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                
