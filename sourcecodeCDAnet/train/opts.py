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
    parser = argparse.ArgumentParser(description='Train Inverse Rayleigh Bernard model')

    # required parameters
    parser.add_argument('--data_folder', type=str, required=True,
                        help='path to data folder')
    parser.add_argument('--train_data', nargs='+', type=str, required=True,
                        help='name of training data filenames')
    parser.add_argument('--eval_data', nargs='+', type=str, required=True,
                        help='name of valid data filenames')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='output folder')
    parser.add_argument('--rayleigh', type=float, required=True,
                        help='Simulation Rayleigh number.')
    parser.add_argument('--prandtl', type=float, required=True,
                        help='Simulation Prandtl number.')
    parser.add_argument('--device', required=True, help='device')


    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size per GPU (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--pseudo_epoch_size', type=int, default=3000, metavar='N',
                        help='number of samples in an pseudo-epoch (default: 3000)')

    parser.add_argument('--lr', type=float, default=1e-2, metavar='R',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=6, metavar='N',
                        help='number of dataloader workers (default: 6)')
    parser.add_argument('--loss_type', default='l1', choices=['l1', 'l2', 'huber'],
                        help='loss type')
    parser.add_argument('--alpha_pde', default=1., type=float, 
                        help='weight of pde residue loss.')
    parser.add_argument('--clip_grad', default=1., type=float,
                        help='clip gradient to this value. large value basically deactivates it.')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint if resume is needed')

    parser.add_argument('--nt_numerical', default=0.05, type=float, help='time step of numerical solution.')
    parser.add_argument('--nx_numerical', default=256., type=float, help='num grid points in x of numerical solution.')
    parser.add_argument('--nz_numerical', default=256., type=float, help='num grid points in z of numerical solution.')
    
    parser.add_argument('--nt', default=8, type=int, help='resolution of clip in t.')
    parser.add_argument('--nx', default=256, type=int, help='resolution of clip in x.')
    parser.add_argument('--nz', default=256, type=int, help='resolution of clip in z.')

    parser.add_argument("--downsamp_t", default=2, type=int,
                        help="down sampling factor in t for low resolution crop.")
    parser.add_argument("--downsamp_xz", default=4, type=int,
                        help="down sampling factor in x and z for low resolution crop.")
    parser.add_argument("--velocityOnly", type=str2bool, nargs='?', default=False, const=True,
                        help='Whether to use Velocity Observations Alone!')

    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per clip.")
    

    # parser.add_argument('--lat_dims', default=32, type=int, help='number of latent dimensions.')
    # parser.add_argument('--unet_nf', default=16, type=int,
    #                     help='number of base number of feature layers in unet.')
    # parser.add_argument('--unet_mf', default=128, type=int,
    #                     help='a cap for max number of feature layers throughout the unet.')
    #                     # 256
    # parser.add_argument('--imnet_nf', default=32, type=int,
    #                     help='number of base number of feature layers in implicit network.')


    parser.add_argument('--lat_dims', default=32, type=int, help='number of latent dimensions.')
    parser.add_argument('--unet_nf', default=16, type=int,
                        help='number of base number of feature layers in unet.')
    parser.add_argument('--unet_mf', default=256, type=int,
                        help='a cap for max number of feature layers throughout the unet.')
                        # 256
    parser.add_argument('--imnet_nf', default=32, type=int,
                        help='number of base number of feature layers in implicit network.')
                        

    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))                    


    parser.add_argument('--nonlin', type=str, default='softplus', choices=list(NONLINEARITIES.keys()),
                        help='Nonlinear activations for continuous decoder.')
    parser.add_argument('--use_continuity', type=str2bool, nargs='?', default=False, const=True,
                        help='Whether to enforce continuity equation (mass conservation) or not')


    parser.add_argument('--normalize_channels', dest='normalize_channels', action='store_true')
    parser.add_argument('--no_normalize_channels', dest='normalize_channels', action='store_false')
    parser.set_defaults(normalize_channels=True)

    parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true')
    parser.add_argument('--no_lr_scheduler', dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)

    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--debug', action='store_true',
                        help='Run with batch_size=4, epochs=3, pseudo_epoch_size=100, log_interval=1')

    args = parser.parse_args()

    if args.debug:
        print('####### DEBUG MODE #######')
        args.batch_size = 4
        args.epochs = 3
        args.pseudo_epoch_size = 100
        args.log_interval = 1

    return args