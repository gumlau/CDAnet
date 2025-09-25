from __future__ import division, print_function

import argparse
from glob import glob

import json
import os
import numpy as np
import random
import time
from collections import defaultdict
from itertools import chain
import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision.datasets.samplers import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# import our modules
import sys

sys.path.append("../model")
import utilities

import dataloader_spacetime as loader
from dataloader_spacetime import RB2DataLoader
from physics import get_rb2_pde_layer

from unet3d import UNet3d
from implicit_net import ImNet
from nonlinearities import NONLINEARITIES

# TO REMOVE
###
from local_implicit_grid import query_local_implicit_grid


###

def fwd_fn(imnet, latent_grid, grid_point_coord):
    # concatenate the grid_point_coord
    latent_grid_with_coord = torch.cat((latent_grid, grid_point_coord), dim=-1)
    output = imnet(latent_grid_with_coord)
    return output


def train_one_epoch(args, unet, imnet, train_loader, epoch, global_step, device, criterion, writer,
                    optimizer, pde_layer):
    unet.train()
    imnet.train()

    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    metric_logger = utilities.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utilities.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = f'Train Epoch {epoch}:'
    for batch_idx, data_tensors in enumerate(
            metric_logger.log_every(train_loader, args.log_interval, header, device=device)):
        # enumerate(train_loader):
        # for batch_idx, (space_time_crop_hres, space_time_crop_lres, grid_point_coord) in enumerate(metric_logger.log_every(train_loader, args.log_interval, header, device=device)):
        start_time = time.time()

        data_tensors = [t.to(device) for t in data_tensors]
        input_grid, point_coord, point_value = data_tensors

        optimizer.zero_grad()

        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(pde_fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True)

        # function value regression loss
        reg_loss = criterion(pred_value, point_value)

        # pde residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        pde_loss = criterion(pde_tensors, torch.zeros_like(pde_tensors))
        loss = reg_loss + args.alpha_pde * pde_loss

        for param in chain(unet.parameters(), imnet.parameters()):
            param.grad = None
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(chain(unet.parameters(), imnet.parameters()), args.clip_grad)

        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(reg_loss=reg_loss.item())
        metric_logger.update(pde_loss=pde_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if batch_idx % args.log_interval == 0 and utilities.is_main_process():
            writer.add_scalar('metrics_per_batch/train/reg_loss', reg_loss.item(), global_step=int(global_step))
            writer.add_scalar('metrics_per_batch/train/pde_loss', pde_loss.item(), global_step=int(global_step))
            writer.add_scalar('metrics_per_batch/train/loss', loss.item(), global_step=int(global_step))

        global_step += 1

    total_loss = metric_logger.meters['loss'].global_avg
    return total_loss


def compute_eval_metrics(args, hres_pred, space_time_crop_hres):
    nBatch, n_ch, n_t, n_z, n_x = space_time_crop_hres.shape

    # print("Pressure")
    # print(hres_pred[0,0,3, 50, 50])
    # print(space_time_crop_hres[0,0,3, 50, 50])

    # print("Temperature")
    # print(hres_pred[0,1,3, 50, 50])
    # print(space_time_crop_hres[0,1,3, 50, 50])

    # print("U-VEl")
    # print(hres_pred[0,2,3, 50, 50])
    # print(space_time_crop_hres[0,2,3, 50, 50])

    # print("V-vel")
    # print(hres_pred[0,3,3, 50, 50])
    # print(space_time_crop_hres[0,3,3, 50, 50])

    difference = hres_pred - space_time_crop_hres

    diffAbs = torch.absolute(difference)
    diffSq = torch.square(difference)

    # Sum over space
    sumX = torch.sum(diffSq, dim=4)
    # sumX_abs = torch.sum(diffAbs, dim=4)
    sumSquared = torch.sum(sumX, dim=3)
    # AE = torch.sum(sumX_abs*(1/(float(n_x)*float(n_z))), dim=3)

    normalization = torch.sqrt(
        torch.sum(torch.sum(torch.square(space_time_crop_hres), dim=4), dim=3) * (1 / (float(n_x) * float(n_z))))
    # normalization_l1 = torch.absolute(torch.sum(torch.sum(space_time_crop_hres, dim=4), dim=3))*(1/(float(n_x)*float(n_z)))
    RMSE = torch.sqrt(sumSquared * (1 / (float(n_x) * float(n_z))))

    RRMSE = RMSE / normalization
    # NMAE = AE/normalization_l1

    # Average over time and batches
    # RRMSE_copy = torch.clone(RRMSE)
    # RRMSE_copy = torch.mean(RRMSE_copy, dim=0)

    RRMSE = torch.mean(RRMSE, dim=2)
    RRMSE = torch.mean(RRMSE, dim=0)

    # AE = torch.mean(AE, dim=2)
    # AE = torch.mean(AE, dim=0)

    # NMAE = torch.mean(NMAE, dim=2)
    # NMAE = torch.mean(NMAE, dim=0)

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
        # 'NMAE_p' : NMAE[0],
        # 'NMAE_T' : NMAE[1],
        # 'NMAE_u' : NMAE[2],
        # 'NMAE_v' : NMAE[3],
    }

    return results


def save_qualitative_results_to_tensorboard(velocityOnly, hres_pred, space_time_crop_hres, space_time_crop_lres, writer,
                                            global_step):
    phys_channels = ['p', 'b', 'u', 'w']
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    # print(velocityOnly)

    if not velocityOnly:
        # SAVE FIRST 3 IMAGES OF THE BATCH
        for sample_id in range(space_time_crop_hres.shape[0]):
            for channel_name, channel_id in phys2id.items():
                pred_tensor = hres_pred[sample_id, channel_id, ...]  # [nt, nz, nx]
                hres_tensor = space_time_crop_hres[sample_id, channel_id, ...]  # [nt, nz, nx]
                lres_tensor = space_time_crop_lres[sample_id, channel_id, ...]  # [nt, nz, nx]

                pred_image = utilities.batch_colorize_scalar_tensors(pred_tensor)  # [nt, nz, nx, 3]
                hres_image = utilities.batch_colorize_scalar_tensors(hres_tensor)  # [nt, nz, nx, 3]
                lres_image = utilities.batch_colorize_scalar_tensors(lres_tensor)  # [nt, nz, nx, 3]

                writer.add_images(f'sample_{sample_id}/{channel_name}/Pred', pred_image,
                                  dataformats='NHWC', global_step=int(global_step))
                writer.add_images(f'sample_{sample_id}/{channel_name}/GT', hres_image,
                                  dataformats='NHWC', global_step=int(global_step))
                writer.add_images(f'sample_{sample_id}/{channel_name}/Input', lres_image,
                                  dataformats='NHWC', global_step=int(global_step))
    else:
        for sample_id in range(space_time_crop_hres.shape[0]):
            # SAVE FIRST 3 IMAGES OF THE BATCH
            for channel_name, channel_id in phys2id.items():
                pred_tensor = hres_pred[sample_id, channel_id, ...]  # [nt, nz, nx]
                hres_tensor = space_time_crop_hres[sample_id, channel_id, ...]  # [nt, nz, nx]
                if channel_id < 2:
                    lres_tensor = space_time_crop_lres[sample_id, channel_id, ...]  # [nt, nz, nx]

                pred_image = utilities.batch_colorize_scalar_tensors(pred_tensor)  # [nt, nz, nx, 3]
                hres_image = utilities.batch_colorize_scalar_tensors(hres_tensor)  # [nt, nz, nx, 3]
                if channel_id < 2:
                    lres_image = utilities.batch_colorize_scalar_tensors(lres_tensor)  # [nt, nz, nx, 3]

                writer.add_images(f'sample_{sample_id}/{channel_name}/Pred', pred_image,
                                  dataformats='NHWC', global_step=int(global_step))
                writer.add_images(f'sample_{sample_id}/{channel_name}/GT', hres_image,
                                  dataformats='NHWC', global_step=int(global_step))
                if channel_id < 2:
                    writer.add_images(f'sample_{sample_id}/{channel_name}/Input', lres_image,
                                      dataformats='NHWC', global_step=int(global_step))

                # def valid_one_epoch2(args, unet, imnet, eval_loader, epoch, global_step, device, criterion, writer, pde_layer):


#     """Eval function. Used for evaluating entire slices and comparing to GT."""
#     unet.eval()
#     imnet.eval()
#     phys_channels = ["p", "b", "u", "w"]
#     phys2id = dict(zip(phys_channels, range(len(phys_channels))))
#     xmin = torch.zeros(3, dtype=torch.float32).to(device)
#     xmax = torch.ones(3, dtype=torch.float32).to(device)

#     metric_logger = utilities.MetricLogger(delimiter="  ")
#     header = f'Valid Epoch {epoch}:'

#     for batch_idx, data_tensors in enumerate(metric_logger.log_every(eval_loader, args.log_interval, header, device=device)):
#         break
#     # for data_tensors in eval_loader:
#     #     # only need the first batch
#     #     break
#     # send tensors to device
#     data_tensors = [t.to(device) for t in data_tensors]
#     hres_grid, lres_grid, _, _ = data_tensors
#     latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
#     nb, nc, nt, nz, nx = hres_grid.shape

#     # permute such that C is the last channel for local implicit grid query
#     latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

#     # define lambda function for pde_layer
#     fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

#     # update pde layer and compute predicted values + pde residues
#     pde_layer.update_forward_method(fwd_fn)

#     # layout query points for the desired slices
#     eps = 1e-6
#     t_seq = torch.linspace(eps, 1-eps, nt)[::int(nt/8)]  # temporal sequences
#     z_seq = torch.linspace(eps, 1-eps, nz)  # z sequences
#     x_seq = torch.linspace(eps, 1-eps, nx)  # x sequences

#     query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
#     query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
#     n_query = query_coord.shape[0]

#     res_dict = defaultdict(list)

#     n_iters = int(np.ceil(n_query/args.pseudo_epoch_size))

#     for idx in range(n_iters):
#         sid = idx * args.pseudo_epoch_size
#         eid = min(sid+args.pseudo_epoch_size, n_query)
#         query_coord_batch = query_coord[sid:eid]
#         query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

#         pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
#         pred_value = pred_value.detach()
#         for key in residue_dict.keys():
#             residue_dict[key] = residue_dict[key].detach()
#         for name, chan_id in zip(phys_channels, range(4)):
#             res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
#         for name, val in residue_dict.items():
#             res_dict[name].append(val[..., 0])   # [b, pb]

#     for key in res_dict.keys():
#         res_dict[key] = (torch.cat(res_dict[key], axis=1)
#                          .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))

#     # log the imgs sample-by-sample
#     for samp_id in range(nb):
#         for key in res_dict.keys():
#             field = res_dict[key][samp_id]  # [nt, nz, nx]
#             # add predicted slices
#             images = utilities.batch_colorize_scalar_tensors(field)  # [nt, nz, nx, 3]

#             writer.add_images('sample_{}/{}/predicted'.format(samp_id, key), images,
#                 dataformats='NHWC', global_step=int(global_step))
#             # add ground truth slices (only for phys channels)
#             if key in phys_channels:
#                 gt_fields = hres_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
#                 gt_images = utilities.batch_colorize_scalar_tensors(gt_fields)  # [nt, nz, nx, 3]

#                 writer.add_images('sample_{}/{}/ground_truth'.format(samp_id, key), gt_images,
#                     dataformats='NHWC', global_step=int(global_step))

#                 input_fields = lres_grid[samp_id, phys2id[key], ::int(nt/8)]  # [nt, nz, nx]
#                 input_images = utilities.batch_colorize_scalar_tensors(input_fields)  # [nt, nz, nx, 3]

#                 writer.add_images('sample_{}/{}/Input'.format(samp_id, key), input_images,
#                     dataformats='NHWC', global_step=int(global_step))

#     total_loss = metric_logger.meters['loss'].global_avg
#     total_eval_metrics = {k: metric_logger.meters[k].global_avg for k in eval_metrics_results.keys()}
#     return total_loss, total_eval_metrics


def valid_one_epoch(args, unet, imnet, eval_loader, epoch, global_step, device, criterion, writer, pde_layer,
                    eval_dataset):
    unet.eval()
    imnet.eval()

    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    metric_logger = utilities.MetricLogger(delimiter="  ")
    # header = f'Valid Epoch {epoch}:'
    # with torch.no_grad():
    # for data_tensors in eval_loader:
    # only need the first batch
    # break
    header = f'Eval Epoch {epoch}:'
    for batch_idx, data_tensors in enumerate(
            metric_logger.log_every(eval_loader, args.log_interval, header, device=device)):
        break
    # for batch_idx, data_tensors in enumerate(eval_loader):
    start_time = time.time()

    data_tensors = [t.to(device) for t in data_tensors]
    hres_grid, lres_grid, _, _ = data_tensors

    latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
    nb, nc, nt, nz, nx = hres_grid.shape

    # permute such that C is the last channel for local implicit grid query
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

    # define lambda function for pde_layer
    pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(pde_fwd_fn)

    # layout query points for the desired slices
    eps = 1e-6
    # t_seq = torch.linspace(eps, 1-eps, nt)[::int(nt/4)]  # temporal sequences
    t_seq = torch.linspace(eps, 1 - eps, nt)  # temporal sequences
    z_seq = torch.linspace(eps, 1 - eps, nz)  # z sequences
    x_seq = torch.linspace(eps, 1 - eps, nx)  # x sequences

    query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
    query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
    n_query = query_coord.shape[0]

    n_iters = int(np.ceil(n_query / args.pseudo_epoch_size))

    res_dict = defaultdict(list)

    for idx in range(n_iters):
        sid = idx * args.pseudo_epoch_size
        eid = min(sid + args.pseudo_epoch_size, n_query)
        query_coord_batch = query_coord[sid:eid]
        query_coord_batch = query_coord_batch[None].expand(*(nb, eid - sid, 3))  # [nb, eid-sid, 3]

        pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        pred_value = pred_value.detach()

        for key in residue_dict.keys():
            residue_dict[key] = residue_dict[key].detach()
        # if idx == 0:
        #     pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        #     for key in residue_dict.keys():
        #         residue_dict[key] = residue_dict[key].detach()
        # else:
        #     pred_value = pde_layer(query_coord_batch, return_residue=False)

        for name, chan_id in zip(phys_channels, range(4)):
            res_dict[name].append(pred_value[..., chan_id])  # [b, pb]
        for name, val in residue_dict.items():
            res_dict[name].append(val[..., 0])  # [b, pb]

    for key in res_dict.keys():
        if key in phys_channels:
            res_dict[key] = (torch.cat(res_dict[key], axis=1)
                             .reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))

    # print("here 1")
    predTMP = torch.stack([res_dict[key] for key in phys_channels], axis=0).to(device)
    # print("here 2")
    predTMP = eval_dataset.denormalize_grid(predTMP)
    # print("here 3")
    predTMP = predTMP.permute(1, 0, 2, 3, 4)

    loss = criterion(predTMP, hres_grid)
    metric_logger.update(loss=loss.item())

    # print("here 4")
    eval_metrics_results = compute_eval_metrics(args, predTMP.detach(), hres_grid.detach())
    metric_logger.update(**eval_metrics_results)

    # if batch_idx % args.log_interval == 0 and utilities.is_main_process():
    #         writer.add_scalar('metrics_per_batch/valid/loss', loss.item(), global_step=int(global_step))
    #         for k, v in eval_metrics_results.items():
    #             writer.add_scalar(f'metrics_per_batch/valid/{k}', v, global_step=int(global_step))

    # log the imgs sample-by-sample
    # if batch_idx == 0:
    # log the imgs sample-by-sample
    for samp_id in range(nb):
        for key in phys_channels:
            field = res_dict[key][samp_id]  # [nt, nz, nx]
            images = utilities.batch_colorize_scalar_tensors(field)  # [nt, nz, nx, 3]

            writer.add_images('sample_{}/{}/predicted'.format(samp_id, key), images,
                              dataformats='NHWC', global_step=int(global_step))

            gt_fields = hres_grid[samp_id, phys2id[key], ::int(nt / 8)]  # [nt, nz, nx]
            gt_images = utilities.batch_colorize_scalar_tensors(gt_fields)  # [nt, nz, nx, 3]

            writer.add_images('sample_{}/{}/ground_truth'.format(samp_id, key), gt_images,
                              dataformats='NHWC', global_step=int(global_step))

            input_fields = lres_grid[samp_id, phys2id[key], ::int(nt / 8)]  # [nt, nz, nx]
            input_images = utilities.batch_colorize_scalar_tensors(input_fields)  # [nt, nz, nx, 3]

            writer.add_images('sample_{}/{}/Input'.format(samp_id, key), input_images,
                              dataformats='NHWC', global_step=int(global_step))

    global_step += 1

    total_loss = metric_logger.meters['loss'].global_avg
    total_eval_metrics = {k: metric_logger.meters[k].global_avg for k in eval_metrics_results.keys()}
    return total_loss, total_eval_metrics


"""
"""

"""
"""


def valid_one_epoch2(args, unet, imnet, eval_loader, epoch, global_step, device, criterion, writer, pde_layer,
                     eval_dataset):
    unet.eval()
    imnet.eval()

    phys_channels = ["p", "b", "u", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    # pred_hres = []

    metric_logger = utilities.MetricLogger(delimiter="  ")
    # header = f'Valid Epoch {epoch}:'
    # with torch.no_grad():
    # for data_tensors in eval_loader:
    # only need the first batch
    # break
    header = f'Eval Epoch {epoch}:'
    with torch.no_grad():
        for batch_idx, data_tensors in enumerate(
                metric_logger.log_every(eval_loader, args.log_interval, header, device=device)):
            start_time = time.time()

            data_tensors = [t.to(device) for t in data_tensors]
            hres_grid, lres_grid, _, _ = data_tensors

            latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
            nb, nc, nt, nz, nx = hres_grid.shape

            # permute such that C is the last channel for local implicit grid query
            latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

            # define lambda function for pde_layer ~ This will be the forward function to push through the NN
            pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

            # update pde layer and compute predicted values + pde residues
            pde_layer.update_forward_method(pde_fwd_fn)

            # layout query points for the desired slices
            eps = 1e-6
            # t_seq = torch.linspace(eps, 1-eps, nt)[::int(nt/4)]  # temporal sequences
            t_seq = torch.linspace(eps, 1 - eps, nt)  # temporal sequences
            z_seq = torch.linspace(eps, 1 - eps, nz)  # z sequences
            x_seq = torch.linspace(eps, 1 - eps, nx)  # x sequences

            query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq), axis=-1)  # [nt, nz, nx, 3]
            query_coord = query_coord.reshape([-1, 3]).to(device)  # [nt*nz*nx, 3]
            n_query = query_coord.shape[0]

            n_iters = int(np.ceil(n_query / args.pseudo_epoch_size))

            res_dict = defaultdict(list)

            for idx in range(n_iters):
                sid = idx * args.pseudo_epoch_size
                eid = min(sid + args.pseudo_epoch_size, n_query)
                query_coord_batch = query_coord[sid:eid]
                query_coord_batch = query_coord_batch[None].expand(*(nb, eid - sid, 3))  # [nb, eid-sid, 3]

                pred_value = pde_layer(query_coord_batch, return_residue=False)
                # print(pred_value)
                # print(type(pred_value))
                pred_value = pred_value.detach()
                # print(pred_value)
                # print(type(pred_value))

                for name, chan_id in zip(phys_channels, range(4)):
                    res_dict[name].append(pred_value[..., chan_id])  # [b,
                # print(res_dict)

            for key in res_dict.keys():
                res_dict[key] = (torch.cat(res_dict[key], axis=1).reshape([nb, len(t_seq), len(z_seq), len(x_seq)]))

            pred_hres = torch.stack([res_dict["p"], res_dict["b"], res_dict["u"], res_dict["w"]], dim=0)
            pred_hres = pred_hres.permute(1, 0, 2, 3, 4)  # [batch, C, T, X, Y]

            loss = criterion(pred_hres, hres_grid)
            metric_logger.update(loss=loss.item())

            eval_metrics_results = compute_eval_metrics(args, pred_hres.detach(), hres_grid.detach())
            metric_logger.update(**eval_metrics_results)

            if batch_idx % args.log_interval == 0 and utilities.is_main_process():
                writer.add_scalar('metrics_per_batch/valid/loss', loss.item(), global_step=int(global_step))
                for k, v in eval_metrics_results.items():
                    writer.add_scalar(f'metrics_per_batch/valid/{k}', v, global_step=int(global_step))

            if batch_idx == 0:
                save_qualitative_results_to_tensorboard(args.velocityOnly, pred_hres, hres_grid, lres_grid, writer,
                                                        global_step)

            global_step += 1

    total_loss = metric_logger.meters['loss'].global_avg
    total_eval_metrics = {k: metric_logger.meters[k].global_avg for k in eval_metrics_results.keys()}
    return total_loss, total_eval_metrics


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

    # random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create dataloaders
    train_dataset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.train_data,
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=False,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp,
        velOnly=args.velocityOnly, normalize_hres=args.normalize_channels
    )
    eval_dataset = loader.RB2DataLoader(
        data_dir=args.data_folder, data_filename=args.eval_data,
        nx=args.nx, nz=args.nz, nt=args.nt, n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels, return_hres=True,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp,
        velOnly=args.velocityOnly, normalize_hres=args.normalize_channels
    )

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(eval_dataset, replacement=True, num_samples=1000)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler, shuffle=True)
        eval_sampler = DistributedSampler(eval_sampler, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                             sampler=eval_sampler, num_workers=args.num_workers, pin_memory=True)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #                         sampler=train_sampler, num_workers=args.num_workers, pin_memory=False)
    # eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #                         sampler=eval_sampler, num_workers=args.num_workers, pin_memory=False)

    # setup model
    # unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=(args.nt, args.nz, args.nx),
    #               nf=args.unet_nf, mf=args.unet_mf)
    # imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
    #               activation=NONLINEARITIES[args.nonlin])

    # torch.set_num_threads(1)

    if args.velocityOnly:
        unet = UNet3d(in_features=2, out_features=args.lat_dims, igres=train_dataset.scale_lres,
                      nf=args.unet_nf, mf=args.unet_mf)
    else:
        unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=train_dataset.scale_lres,
                      nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])

    unet.to(device)
    imnet.to(device)

    unet_without_ddp = unet
    imnet_without_ddp = imnet
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[args.gpu])
        unet_without_ddp = unet.module
        imnet = torch.nn.parallel.DistributedDataParallel(imnet, device_ids=[args.gpu])
        imnet_without_ddp = imnet.module

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    print(f'{model_param_count(unet)}(unet) + {model_param_count(imnet)}(imnet) parameters in total')

    # optimizer
    lr = args.lr * args.world_size
    optimizer = torch.optim.SGD(
        params=chain(unet.parameters(), imnet.parameters()),
        lr=lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.loss_type == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss_type == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    start_ep = 0
    train_global_step = np.zeros(1, dtype=np.uint32)
    valid_global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        print(f'Resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_ep = checkpoint['epoch']
        train_global_step = checkpoint['train_global_step']
        valid_global_step = checkpoint['valid_global_step']
        tracked_stats = checkpoint['tracked_stats']
        unet_without_ddp.load_state_dict(checkpoint['unet_state_dict'])
        imnet_without_ddp.load_state_dict(checkpoint['imnet_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = train_dataset.channel_mean
        std = train_dataset.channel_std
    else:
        mean = std = None
    pde_layer = get_rb2_pde_layer(mean=mean, std=std,
                                  t_crop=args.nt * args.nt_numerical, z_crop=args.nz * (1. / args.nz_numerical),
                                  x_crop=args.nx * (1. / args.nx_numerical),
                                  prandtl=args.prandtl, rayleigh=args.rayleigh,
                                  use_continuity=args.use_continuity)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            eval_sampler.set_epoch(epoch)  # changed from eval_sampler to eval_sampler

        train_loss = train_one_epoch(args, unet, imnet, train_loader,
                                     epoch, train_global_step, device, criterion, writer, optimizer, pde_layer)
        valid_loss, eval_metrics = valid_one_epoch2(args, unet, imnet, eval_loader,
                                                    epoch, valid_global_step, device, criterion, writer, pde_layer,
                                                    eval_dataset)
        # valid_loss, eval_metrics = valid_one_epoch2(args, unet, imnet, eval_loader,
        #     epoch, valid_global_step, device, criterion, writer, pde_layer)
        # valid_loss = eval(args, unet, imnet, eval_loader,
        #     epoch, valid_global_step, device, criterion, writer, pde_layer, grid_point_coord)

        print(f'** EPOCH {epoch}: train_loss {train_loss:.2e}; '
              f'valid_loss {valid_loss:.2e}; eval_metrics {eval_metrics}')

        writer.add_scalar('metrics_per_epoch/train_loss', train_loss, global_step=epoch)
        writer.add_scalar('metrics_per_epoch/valid_loss', valid_loss, global_step=epoch)
        writer.add_scalar('metrics_per_epoch/lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        for k, v in eval_metrics.items():
            writer.add_scalar(f'metrics_per_epoch/{k}', v, global_step=epoch)

        if args.lr_scheduler:
            scheduler.step(valid_loss)

        # is_best = valid_loss < tracked_stats
        # tracked_stats = min(valid_loss, tracked_stats)

        # is_best = 0.5*(eval_metrics['RRMSE_u']+ eval_metrics['RRMSE_v']) < tracked_stats
        # tracked_stats = min(0.5*(eval_metrics['RRMSE_u']+ eval_metrics['RRMSE_v']), tracked_stats)

        is_best = eval_metrics['RRMSE_T'] < tracked_stats
        tracked_stats = min(eval_metrics['RRMSE_T'], tracked_stats)

        checkpoint = {
            'args': args,
            'epoch': epoch,
            'unet_state_dict': unet_without_ddp.state_dict(),
            'imnet_state_dict': imnet_without_ddp.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'tracked_stats': tracked_stats,
            'train_global_step': train_global_step,
            'valid_global_step': valid_global_step,
        }

        utilities.save_on_master(
            checkpoint,
            os.path.join(args.output_folder, f'checkpoint_latest.pth'))
        if is_best:
            print(f'Got better valid loss {valid_loss:.2e}. Saving new best model')
            utilities.save_on_master(
                checkpoint,
                os.path.join(args.output_folder, f'checkpoint_best.pth'))


if __name__ == '__main__':
    from opts import parse_args

    args = parse_args()
    main(args)

