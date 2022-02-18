import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver, moving_average
import argparse
import wandb

from core.data_loader import InputFetcher
import time
import datetime
import core.utils as utils
from metrics.eval import calculate_metrics
from PIL import Image


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


args = argparse.Namespace(batch_size=8, beta1=0.0, beta2=0.99,\
                          checkpoint_dir='expr/checkpoints', ds_iter=100000,\
                          eval_dir='expr/eval', eval_every=50000, f_lr=1e-06,\
                          hidden_dim=512, img_size=256, inp_dir='assets/representative/custom/female',\
                          lambda_cyc=1, lambda_ds=1, lambda_reg=1, lambda_sty=1, latent_dim=16,\
                          lm_path='expr/checkpoints/celeba_lm_mean.npz', lr=0.0001, mode='train',\
                          num_domains=2, num_outs_per_domain=10, num_workers=4, \
                          out_dir='assets/representative/celeba_hq/src/female', print_every=10,\
                          randcrop_prob=0.5, ref_dir='assets/representative/celeba_hq/ref',\
                          result_dir='expr/results', resume_iter=0, sample_dir='expr/samples',\
                          sample_every=5000, save_every=10000, seed=777,\
                          src_dir='assets/representative/celeba_hq/src', style_dim=64, \
                          total_iters=100000, train_img_dir='data/celeba_hq/train', val_batch_size=32,\
                          val_img_dir='data/celeba_hq/val', w_hpf=1, weight_decay=0.0001,\
                          wing_path='expr/checkpoints/wing.ckpt')


# ARGS for Celeba HQ
args.mode = 'eval'
args.num_domains = 2
args.w_hpf = 1
args.resume_iter = 100000
args.train_img_dir='data/celeba_hq/train'
args.val_img_dir ='data/celeba_hq/val'
args.checkpoint_dir= 'expr/checkpoints/celeba_hq'
args.eval_dir='expr/eval/celeba_hq'


# LOAD TEACHER
args.model_type = 'teacher'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(args)
cudnn.benchmark = True
torch.manual_seed(args.seed)
solver = Solver(args)
teacher = solver.nets_ema
resume_iter = args.resume_iter
solver._load_checkpoint(args.resume_iter)


# LOAD STUDENT on Celeba HQ
args.model_type = 'student'
args.mode = 'train'
args.resume_iter = 0
args.num_domains = 2
args.w_hpf = 1
args.lambda_reg = 1
args.lambda_sty = 1
args.lambda_ds = 1
args.lambda_cyc = 1
args.train_img_dir = 'data/celeba_hq/train'
args.val_img_dir = 'data/celeba_hq/val'
args.checkpoint_dir = 'expr/checkpoints/celeba_hq_student'
args.eval_dir = 'expr/eval/celeba_hq_student'
args.sample_every = 1000
args.eval_every = 10000
args.save_every = 10000
solver = Solver(args)

# WANDB
wandb.init(project="distill-stargan-v2")
args.val_batch_size=4 # for sampling

if args.mode == 'train':
    assert len(subdirs(args.train_img_dir)) == args.num_domains
    assert len(subdirs(args.val_img_dir)) == args.num_domains
    loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                         which='source',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         prob=args.randcrop_prob,
                                         num_workers=args.num_workers),
                    ref=get_train_loader(root=args.train_img_dir,
                                         which='reference',
                                         img_size=args.img_size,
                                         batch_size=args.batch_size,
                                         prob=args.randcrop_prob,
                                         num_workers=args.num_workers),
                    val=get_test_loader(root=args.val_img_dir,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers))


lambda_mse = 1

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    if z_trgs is not None:
        s_trg = teacher.mapping_network(z_trg, y_trg)
        # s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = teacher.style_encoder(x_ref, y_trg)
        # s_trg = nets.style_encoder(x_ref, y_trg)

    # mse loss
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    t_fake = teacher.generator(x_real, s_trg, masks=masks)
    loss_mse = ((t_fake - x_fake)**2).mean()

    loss = lambda_mse * loss_mse
    return loss, Munch(mse=loss_mse.item())


# DISTILL SCRIPT
nets = solver.nets
nets_ema = solver.nets_ema
optims = solver.optims

# fetch random validation images for debugging
fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
inputs_val = next(fetcher_val)

# resume training if necessary
if args.resume_iter > 0:
    solver._load_checkpoint(args.resume_iter)

# remember the initial value of ds weight
initial_lambda_ds = args.lambda_ds

print('Start training...')
start_time = time.time()
for i in range(args.resume_iter, args.total_iters):
    # fetch images and labels
    inputs = next(fetcher)
    x_real, y_org = inputs.x_src, inputs.y_src
    x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
    z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

    masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

    # train the generator
    g_loss, g_losses_latent = compute_g_loss(
        nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
    solver._reset_grad()
    g_loss.backward()
    optims.generator.step()
    optims.mapping_network.step()
    optims.style_encoder.step()

    g_loss, g_losses_ref = compute_g_loss(
        nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
    solver._reset_grad()
    g_loss.backward()
    optims.generator.step()

    # compute moving average of network parameters
    moving_average(nets.generator, nets_ema.generator, beta=0.999)
    moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
    moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

    # decay weight for diversity sensitive loss
    if args.lambda_ds > 0:
        args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

    # print out log info
    if (i+1) % args.print_every == 0:
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
        all_losses = dict()
        for loss, prefix in zip([g_losses_latent, g_losses_ref],
                                ['G/latent_', 'G/ref_']):
            
            for key, value in loss.items():
                all_losses[prefix + key] = value

        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
        print(log)
        
        for key, value in all_losses.items():
            wandb.log({key: value})

        
    # generate images for debugging
    args.num_outs_per_domain=2 # for sampling
    if (i+1) % args.sample_every == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        lst_filename = utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)
        
        wandb.log({"samples/cycle_consistency": [wandb.Image(Image.open(lst_filename[0]),
                                                             caption=f"filename={lst_filename[0][13:]}")]})
        wandb.log({"samples/latent": [wandb.Image(Image.open(lst_filename[1]),
                                                             caption=f"filename={lst_filename[1][13:]}")]})
        wandb.log({"samples/reference": [wandb.Image(Image.open(lst_filename[2]),
                                                             caption=f"filename={lst_filename[2][13:]}")]})
        
    # save model checkpoints  
    if (i+1) % args.save_every == 0:
        solver._save_checkpoint(step=i+1)

    # compute FID and LPIPS if necessary
    args.val_batch_size = 32  # for evaluating
    args.num_outs_per_domain = 10  # было 2 для быстрого тестирования
    if (i+1) % args.eval_every == 0:
        lpips_latent, fid_latent = calculate_metrics(nets_ema, args, i+1, mode='latent')
        lpips_reference, fid_reference = calculate_metrics(nets_ema, args, i+1, mode='reference')
        
        for key, value in lpips_latent.items():
            wandb.log({key: value})

        for key, value in fid_latent.items():
            wandb.log({key: value})

        for key, value in lpips_reference.items():
            wandb.log({key: value})

        for key, value in fid_reference.items():
            wandb.log({key: value})
