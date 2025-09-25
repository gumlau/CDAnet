from __future__ import division, print_function

import json
import os
import numpy as np
import random
import time
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets.samplers import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from itertools import chain

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import shutil
import os
#import our modules
import sys
sys.path.append("../train")
sys.path.append("../model")
from unet3d import UNet3d
from implicit_net import ImNet
from pde import PDELayer
from nonlinearities import NONLINEARITIES
from local_implicit_grid import query_local_implicit_grid

import dataloader_spacetime as loader
from dataloader_spacetime import RB2DataLoader

from physics import get_rb2_pde_layer
from torch_flow_stats import *
import utilities
import sys



def fwd_fn(imnet, latent_grid, grid_point_coord):
    # concatenate the grid_point_coord
    latent_grid_with_coord = torch.cat((latent_grid, grid_point_coord), dim=-1)
    output = imnet(latent_grid_with_coord)
    return output


def compute_eval_metrics(args, hres_pred, space_time_crop_hres):

    nBatch, n_ch, n_t, n_z, n_x = space_time_crop_hres.shape

    difference = hres_pred - space_time_crop_hres

    diffAbs = torch.absolute(difference)
    diffSq = torch.square(difference)

    # Sum over space
    sumX = torch.sum(diffSq, dim=4)
    sumX_abs = torch.sum(diffAbs, dim=4)
    sumSquared = torch.sum(sumX, dim=3)
    AE = torch.sum(sumX_abs*(1/(float(n_x)*float(n_z))), dim=3)

    normalization = torch.sqrt(torch.sum(torch.sum(torch.square(space_time_crop_hres), dim=4), dim=3)*(1/(float(n_x)*float(n_z))))
    normalization_l1 = torch.absolute(torch.sum(torch.sum(space_time_crop_hres, dim=4), dim=3))*(1/(float(n_x)*float(n_z)))
    RMSE = torch.sqrt(sumSquared*(1/(float(n_x)*float(n_z))))

    RRMSE = RMSE/normalization
    NMAE = AE/normalization_l1

    # Average over time and batches
    # RRMSE_copy = torch.clone(RRMSE)
    # RRMSE_copy = torch.mean(RRMSE_copy, dim=0)

    RRMSE = torch.mean(RRMSE, dim=2)
    RRMSE = torch.mean(RRMSE, dim=0)

    # AE = torch.mean(AE, dim=2)
    # AE = torch.mean(AE, dim=0)

    NMAE = torch.mean(NMAE, dim=2)
    NMAE = torch.mean(NMAE, dim=0)

    # RMSE = torch.mean(RMSE, dim=2)
    # RMSE = torch.mean(RMSE, dim=0)

    results = {
        # 'RMSE_p' : RMSE[0],
        # 'RMSE_T' : RMSE[1],
        # 'RMSE_u' : RMSE[2],
        # 'RMSE_v' : RMSE[3],
        'RRMSE_p': RRMSE[0],
        'RRMSE_T': RRMSE[1],
        'RRMSE_u': RRMSE[2],
        'RRMSE_v': RRMSE[3],
        'NMAE_p' : NMAE[0],
        'NMAE_T' : NMAE[1],
        'NMAE_u' : NMAE[2],
        'NMAE_v' : NMAE[3],
    }

    np.savez_compressed(args.save_path+'RMSE_AE_RRMSE', RMSE=RMSE, AE=AE, RRMSE=RRMSE)

    return results


def evaluate_feat_grid(pde_layer, latent_grid, t_seq, z_seq, x_seq, mins, maxs, pseudo_batch_size):
    """Evaluate latent feature grid at fixed intervals.

    Args:
        pde_layer: PDELayer instance where fwd_fn has been defined.
        latent_grid: latent feature grid of shape [batch, T, Z, X, C]
        t_seq: flat torch array of t-coordinates to evaluate
        z_seq: flat torch array of z-coordinates to evaluate
        x_seq: flat torch array of x-coordinates to evaluate
        mins: flat torch array of len 3 for min coords of t, z, x
        maxs: flat torch array of len 3 for max coords of t, z, x
        pseudo_batch_size, int, size of pseudo batch during eval
    Returns:
        res_dict: result dict.
    """
    device = latent_grid.device
    nb = latent_grid.shape[0]
    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))

    query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]

    nt, nz, nx, _ = query_coord.shape
    query_coord = query_coord.reshape([-1, 3]).to(device)
    n_query  = query_coord.shape[0]

    res_dict = defaultdict(list)

    n_iters = int(np.ceil(n_query/pseudo_batch_size))

    with torch.no_grad():
        for idx in tqdm(range(n_iters)):
            sid = idx * pseudo_batch_size
            eid = min(sid+pseudo_batch_size, n_query)
            query_coord_batch = query_coord[sid:eid]
            query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

            pred_value = pde_layer(query_coord_batch, return_residue=False)
            pred_value = pred_value.detach()

            for name, chan_id in zip(phys_channels, range(4)):
                    res_dict[name].append(pred_value[..., chan_id])

        for key in res_dict.keys():
                res_dict[key] = (torch.cat(res_dict[key], axis=1).reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))[0]

        return res_dict


def frames_to_video(frames_pattern, save_video_to, frame_rate=10, keep_frames=False):
    """Create video from frames.

    frames_pattern: str, glob pattern of frames.
    save_video_to: str, path to save video to.
    keep_frames: bool, whether to keep frames after generating video.
    """
    cmd = ("ffmpeg -framerate {frame_rate} -pattern_type glob -i '{frames_pattern}' "
           "-c:v libx264 -r 30 -pix_fmt yuv420p {save_video_to}"
           .format(frame_rate=frame_rate, frames_pattern=frames_pattern,
                   save_video_to=save_video_to))
    os.system(cmd)
    # print
    print("Saving videos to {}".format(save_video_to))
    # delete frames if keep_frames is not needed
    # if not keep_frames:
    #     frames_dir = os.path.dirname(frames_pattern)
    #     shutil.rmtree(frames_dir)


def export_video(args, res_dict, hres, lres, dataset):
    """Export inference result as a video.
    """
    phys_channels = ["p", "b", "u", "w"]
    if dataset:
        hres = dataset.denormalize_grid(hres.copy())
        lres = dataset.denormalize_grid(lres.copy())
        pred = torch.stack([res_dict[key] for key in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)
        # np.savez_compressed(args.save_path+'highres_lowres_pred', hres=hres, lres=lres, pred=pred)

    os.makedirs(args.save_path, exist_ok=True)
    # enumerate through physical channels first

    for idx, name in enumerate(phys_channels):
        frames_dir = os.path.join(args.save_path, f'frames_{name}')
        os.makedirs(frames_dir, exist_ok=True)
        hres_frames = hres[idx]
        lres_frames = lres[idx]
        pred_frames = pred[idx]

        # loop over each timestep in pred_frames
        max_val = np.max(hres_frames)
        min_val = np.min(hres_frames)

        for pid in range(pred_frames.shape[0]):
            hid = int(np.round(pid / (pred_frames.shape[0] - 1) * (hres_frames.shape[0] - 1)))
            lid = int(np.round(pid / (pred_frames.shape[0] - 1) * (lres_frames.shape[0] - 1)))

            fig, axes = plt.subplots(3, figsize=(10, 10))#, 1, sharex=True)
            # high res ground truth
            im0 = axes[0].imshow(hres_frames[hid], cmap='RdBu',interpolation='spline16')
            axes[0].set_title(f'{name} channel, high res ground truth.')
            im0.set_clim(min_val, max_val)
            # low res input
            im1 = axes[1].imshow(lres_frames[lid], cmap='RdBu',interpolation='none')
            axes[1].set_title(f'{name} channel, low  res ground truth.')
            im1.set_clim(min_val, max_val)
            # prediction
            im2 = axes[2].imshow(pred_frames[pid].cpu(), cmap='RdBu',interpolation='spline16')
            axes[2].set_title(f'{name} channel, predicted values.')
            im2.set_clim(min_val, max_val)
            # add shared colorbar
            cbaxes = fig.add_axes([0.1, 0, .82, 0.05])
            fig.colorbar(im2, orientation="horizontal", pad=0, cax=cbaxes)
            frame_name = 'frame_{:03d}.png'.format(pid)
            fig.savefig(os.path.join(frames_dir, frame_name))

        # stitch frames into video (using ffmpeg)
        frames_to_video(
            frames_pattern=os.path.join(frames_dir, "*.png"),
            save_video_to=os.path.join(args.save_path, f"video_{name}.mp4"),
            frame_rate=args.frame_rate, keep_frames=args.keep_frames)



def model_inference(args, lres, pde_layer):
    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct model
    print(f"Loading model parameters from {args.ckpt}...")
    igres = (int(args.nt/args.downsamp_t),
             int(args.nz/args.downsamp_xz),
             int(args.nx/args.downsamp_xz),)
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=igres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])

    # load model params
    resume_dict = torch.load(args.ckpt)
    unet.load_state_dict(resume_dict["unet_state_dict"])
    imnet.load_state_dict(resume_dict["imnet_state_dict"])

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    # evaluate
    latent_grid = unet(torch.tensor(lres, dtype=torch.float32)[None].to(device))
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

    # create evaluation grid
    t_max = float(args.eval_tres/args.nt)
    z_max = 1
    x_max = 3

    # layout query points for the desired slices
    eps = 1e-6
    t_seq = torch.linspace(eps, t_max-eps, args.eval_tres)  # temporal sequences
    z_seq = torch.linspace(eps, z_max-eps, args.eval_zres)  # z sequences
    x_seq = torch.linspace(eps, x_max-eps, args.eval_xres)  # x sequences

    mins = torch.zeros(3, dtype=torch.float32, device=device)
    maxs = torch.tensor([t_max, z_max, x_max], dtype=torch.float32, device=device)

    # define lambda function for pde_layer
    fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, mins, maxs)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(fwd_fn)

    res_dict = evaluate_feat_grid(pde_layer, latent_grid, t_seq, z_seq, x_seq, mins, maxs,
                                  args.eval_pseudo_batch_size)

    return res_dict



def compute_SkillScores(args, unet, imnet, hres, lres, dataset, pde_layer, TF_Ensembles=False):
    # Start by evaluating the model for multiple lres inputs 
    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_channels = ["p", "b", "u", "w"]
    n_ch, n_t, n_z, n_x = lres.shape
    n_chHR, n_tHR, n_zHR, n_xHR = hres.shape

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    # Keep copy of original lres observations

    lresOG = np.copy(lres)
    pred = np.zeros((args.Nens, n_chHR, n_tHR, n_zHR, n_xHR))

    # GENERATE THE LRES PERTURBED OBSERVATION
    # THIS IS DONE SEQUENTIALLY BECAUSE THE UNET IS EVALUATED AS SO... :(
    ###########################################################################
    for i  in range(args.Nens):
        # Get modified lres
        # nb sigma * np.random.randn(...) + mu 
        # mu -> mean
        # sigma -> std dev.

        lres = np.copy(lresOG)
        lres[1,:,:,:] = lres[1,:,:,:] + args.noiseTemp * np.random.randn(n_t, n_z, n_x)
        lres[2,:,:,:] = lres[2,:,:,:] + args.noiseVels * np.random.randn(n_t, n_z, n_x)
        lres[3,:,:,:] = lres[3,:,:,:] + args.noiseVels * np.random.randn(n_t, n_z, n_x)

        # evaluate
        latent_grid = unet(torch.tensor(lres, dtype=torch.float32)[None].to(device))
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

        # create evaluation grid
        t_max = float(args.eval_tres/args.nt)
        z_max = 1
        x_max = 3

        # layout query points for the desired slices
        eps = 1e-6 #1e-6
        t_seq = torch.linspace(eps, t_max-eps, args.eval_tres)  # temporal sequences
        z_seq = torch.linspace(eps, z_max-eps, args.eval_zres)  # z sequences
        x_seq = torch.linspace(eps, x_max-eps, args.eval_xres)  # x sequences

        mins = torch.zeros(3, dtype=torch.float32, device=device)
        maxs = torch.tensor([t_max, z_max, x_max], dtype=torch.float32, device=device)

        # define lambda function for pde_layer ~ This will be the forward function to push through the NN
        pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, mins, maxs)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(pde_fwd_fn)

        res_dict = evaluate_feat_grid(pde_layer, latent_grid, t_seq, z_seq, x_seq, mins, maxs,
                                    args.eval_pseudo_batch_size)

        # if i==0:
        #     export_video(args, res_dict, hres, lres, dataset)


        # lres = dataset.denormalize_grid(lres.copy())
        print(lres.shape)
        predTMP = torch.stack([res_dict[key] for key in phys_channels], axis=0)
        # predTMP = dataset.denormalize_grid(predTMP)
        print(predTMP.shape)

        # TEST IF THIS:
        pred[i,:,:,:,:] = predTMP.cpu()

        # np.savez_compressed(args.save_path+'highres_pred'+str(i), hres=hres, pred=pred[:, :, :, :, :])
    # print(pred.shape)
    # pred of size [Nens, n_channels, n_t, n_]

    # np.savez_compressed(args.save_path+'highres_pred', hres=hres, pred=pred[:, :, n_tHR-1, :, :])

    # pred = np.transpose(pred)
    print(pred.shape)
    difference = pred - hres

    diffAbs = np.absolute(difference)
    diffSq = np.square(difference)
    sumX = np.sum(diffSq, axis=4)
    sumX_abs = np.sum(diffAbs, axis=4)
    sumSquared = np.sum(sumX, axis=3)
    # AE = np.sum(sumX_abs, axis=3)

    tmp = np.sum(np.sum(np.square(hres), axis=3), axis=2)
    normalization = np.sqrt(tmp)*(1/(float(n_xHR)*float(n_zHR)))

    RMSE = np.sqrt(sumSquared)*(1/(float(n_xHR)*float(n_zHR)))
    AE = np.sum(sumX_abs, axis=3)*(1/(float(n_xHR)*float(n_zHR)))

    RRMSE = RMSE/normalization

    np.savez_compressed(args.save_path+'RMSE_AE_RRMSE', RMSE=RMSE, AE=AE, RRMSE=RRMSE)

    return 0


def main(args):
    utilities.init_distributed_mode(args)
    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    device = torch.device(args.device)
    os.makedirs(args.output_folder, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_folder, 'tensorboard'))

    print(f'loading checkpoint {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_args = checkpoint['args']
    print(checkpoint_args)

    # random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create dataloaders
    valid_dataset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.eval_dataset,
        nx=args.eval_xres, nz=args.eval_zres, nt=args.eval_tres, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=True,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp,
        velOnly=args.velocityOnly, normalize_hres=args.normalize_channels
    )


    # print(valid_dataset.dtype)
    hres, lres, _, _ = valid_dataset[0]
    print(lres.dtype)
    print(lres.shape)
    print(hres.shape)


    print(f"Loading model parameters from {args.checkpoint}...")
    igres = (int(args.nt/args.downsamp_t),
             int(args.nz/args.downsamp_xz),
             int(args.nx/args.downsamp_xz),)
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=igres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])


    # setup model
    resume_dict = torch.load(args.checkpoint)
    unet.load_state_dict(resume_dict["unet_state_dict"])
    imnet.load_state_dict(resume_dict["imnet_state_dict"])

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    print(f'{model_param_count(unet)}(unet) + {model_param_count(imnet)}(imnet) parameters in total')

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = valid_dataset.channel_mean
        std = valid_dataset.channel_std
    else:
        mean = std = None
    pde_layer = get_rb2_pde_layer(mean=mean, std=std, prandtl=args.prandtl, rayleigh=args.rayleigh)

    compute_SkillScores(args, unet, imnet, hres, lres, valid_dataset, pde_layer, TF_Ensembles=False)

    # save video
    # export_video(args, res_dict, hres, lres, dataset)

    # eval_metrics = eval(args, unet, imnet, valid_loader,
    #     valid_global_step, device, writer)

    # print(f'** eval_metrics {eval_metrics}')

    # for k, v in eval_metrics.items():
    #     writer.add_scalar(f'metrics_per_epoch/{k}', v, global_step=0)

    writer.flush()
    writer.close()

if __name__ == '__main__':
    from opts import parse_args
    args = parse_args()
    main(args)