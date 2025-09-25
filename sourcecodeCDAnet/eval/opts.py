import argparse
import sys
import torch
sys.path.append('../model')
from nonlinearities import NONLINEARITIES



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    # Evaluation settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('--data_folder', type=str, required=True,
                        help='path to data folder')
    parser.add_argument("--eval_xres", type=int, default=384, metavar="X",
                        help="x resolution during evaluation (default: 384)")
    parser.add_argument("--eval_zres", type=int, default=256, metavar="Z",
                        help="z resolution during evaluation (default: 128)")
    parser.add_argument("--eval_tres", type=int, default=350, metavar="T",
                        help="t resolution during evaluation (default: 350)")



    parser.add_argument("--eval_downsamp_t", default=2, type=int, required=True, 
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--eval_downsamp_xz", default=4, type=int, required=True,
                        help="down sampling factor in x and z for low resolution crop.")

    parser.add_argument("--downsamp_t", default=2, type=int,    
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--downsamp_xz", default=4, type=int,
                        help="down sampling factor in x and z for low resolution crop.")


    parser.add_argument('--checkpoint', type=str, required=True, help="path to checkpoint")
    parser.add_argument("--save_path", type=str, default='eval')
    parser.add_argument("--eval_dataset", nargs='+', type=str, required=True)


    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=7, metavar='N',
                        help='input batch size per GPU (default: 7)')           
    parser.add_argument('--num_workers', type=int, default=6, metavar='N',
                        help='number of dataloader workers (default: 6)')


    parser.add_argument('--pseudo_epoch_size', type=int, default=3000, metavar='N',
                        help='number of samples in an pseudo-epoch (default: 3000)')


    parser.add_argument("--lres_interp", type=str, default='linear',
                        help="str, interpolation scheme for generating low res. choices of 'linear', 'nearest'")
    parser.add_argument("--lres_filter", type=str, default='none',
                        help=" str, filter to apply on original high-res image before \
                        interpolation. choices of 'none', 'gaussian', 'uniform', 'median', 'maximum'")


    parser.add_argument("--frame_rate", type=int, default=10, metavar="N",
                        help="frame rate for output video (default: 10)")

    parser.add_argument("--keep_frames", dest='keep_frames', action='store_true')
    parser.add_argument("--no_keep_frames", dest='keep_frames', action='store_false')

    parser.add_argument("--eval_pseudo_batch_size", type=int, default=10000,
                        help="psudo batch size for querying the grid. set to a smaller"
                             " value if OOM error occurs")
    
    parser.add_argument('--rayleigh', type=float, required=True,
                        help='Simulation Rayleigh number.')
    parser.add_argument('--prandtl', type=float, required=True,
                        help='Simulation Prandtl number.')

    parser.add_argument('--normalize_channels', dest='normalize_channels', action='store_true')
    parser.add_argument('--no_normalize_channels', dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)          
    parser.add_argument("--velocityOnly", type=str2bool, nargs='?', default=False, const=True,
                        help='Whether to use Velocity Observations Alone!')          

    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per clip.")
    
    parser.add_argument('--ensembles', type=str, required=True, default="false", help="Run for Ensembles?")
    parser.add_argument('--Nens', type=int, required=True, default=50, help="Number of ensemble members")
    parser.add_argument('--noiseTemp', type=float, required=True, default=0.1, help="Temperature noise level")
    parser.add_argument('--noiseVels', type=float, required=True, default=0.05, help="Velocity components noise levels")
    
    # Same as Training...
    parser.add_argument('--nt', default=8, type=int, help='resolution of clip in t.')
    parser.add_argument('--nx', default=256, type=int, help='resolution of clip in x.')
    parser.add_argument('--nz', default=256, type=int, help='resolution of clip in z.')
    
    parser.add_argument('--lat_dims', default=32, type=int, help='number of latent dimensions.')
    parser.add_argument('--unet_nf', default=16, type=int,
                        help='number of base number of feature layers in unet.')
    parser.add_argument('--unet_mf', default=128, type=int,
                        help='a cap for max number of feature layers throughout the unet.')
    parser.add_argument('--imnet_nf', default=32, type=int,
                        help='number of base number of feature layers in implicit network.')

    parser.add_argument('--nonlin', type=str, default='softplus', choices=list(NONLINEARITIES.keys()),
                        help='Nonlinear activations for continuous decoder.')

    parser.add_argument('--device', required=True, help='device')         
    parser.add_argument('--output_folder', type=str, required=True,
                        help='output folder')               
    
    parser.set_defaults(keep_frames=False)
    args = parser.parse_args()

    return args