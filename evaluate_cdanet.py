#!/usr/bin/env python
"""
Evaluation script for trained CDAnet models.

Usage:
    python evaluate_cdanet.py --checkpoint checkpoints/best_model.pth --output_dir eval_results
    python evaluate_cdanet.py --checkpoint best_model.pth --generalization --test_ra 7e5 8e5 9e5
"""

import argparse
import os
import torch
import numpy as np

from cdanet.models import CDAnet
from cdanet.config import ExperimentConfig
from cdanet.data import RBDataModule
from cdanet.evaluation import CDAnetEvaluator, evaluate_model_from_checkpoint
from cdanet.utils import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained CDAnet model')
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./rb_data_numerical',
                       help='Directory containing test data')
    parser.add_argument('--config', type=str, help='Configuration file (will try to load from checkpoint if not provided)')
    
    # Evaluation settings
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--splits', nargs='+', default=['test'], choices=['train', 'val', 'test'],
                       help='Dataset splits to evaluate on')
    
    # Generalization testing
    parser.add_argument('--generalization', action='store_true',
                       help='Test generalization to different Ra numbers')
    parser.add_argument('--test_ra', nargs='+', type=float,
                       default=[7e5, 8e5, 9e5, 1.1e6, 1.2e6, 1.5e6, 2e6],
                       help='Ra numbers for generalization testing')
    
    # Analysis options
    parser.add_argument('--temporal_evolution', action='store_true',
                       help='Analyze temporal evolution of predictions')
    parser.add_argument('--physics_analysis', action='store_true',
                       help='Perform detailed physics-based analysis')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save raw predictions for further analysis')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, config_path: str = None) -> tuple:
    """Load model and configuration from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig()
        config.update_from_dict(config_dict)
    elif 'config' in checkpoint:
        config = ExperimentConfig()
        config.update_from_dict(checkpoint['config'])
    else:
        print("Warning: Using default configuration - results may be suboptimal")
        config = ExperimentConfig()
        if 'model_config' in checkpoint:
            config.model.in_channels = checkpoint['model_config'].get('in_channels', config.model.in_channels)
            config.model.feature_channels = checkpoint['model_config'].get('feature_channels', config.model.feature_channels)
            config.model.mlp_hidden_dims = checkpoint['model_config'].get('mlp_hidden_dims', config.model.mlp_hidden_dims)
            config.model.activation = checkpoint['model_config'].get('activation', config.model.activation)
            config.model.coord_dim = checkpoint['model_config'].get('coord_dim', config.model.coord_dim)
            config.model.output_dim = checkpoint['model_config'].get('output_dim', config.model.output_dim)
            config.model.igres = tuple(checkpoint['model_config'].get('igres', config.model.igres))
            config.model.unet_nf = checkpoint['model_config'].get('unet_nf', config.model.unet_nf)
            config.model.unet_mf = checkpoint['model_config'].get('unet_mf', config.model.unet_mf)
            config.model.imnet_nf = checkpoint['model_config'].get('imnet_nf', config.model.imnet_nf)
    
    # Create model
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation,
        coord_dim=config.model.coord_dim,
        output_dim=config.model.output_dim,
        igres=getattr(config.model, 'igres', (4, 64, 64)),
        unet_nf=getattr(config.model, 'unet_nf', 16),
        unet_mf=getattr(config.model, 'unet_mf', 256),
        imnet_nf=getattr(config.model, 'imnet_nf', None)
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'unet_state_dict' in checkpoint and 'imnet_state_dict' in checkpoint:
        model.feature_extractor.load_state_dict(checkpoint['unet_state_dict'])
        model.implicit_net.load_state_dict(checkpoint['imnet_state_dict'])
    else:
        raise KeyError('Checkpoint missing model weights: expected model_state_dict or unet/imnet state dicts')

    return model, config, checkpoint


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CDAnet Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Load model and configuration
    print("Loading model and configuration...")
    model, config, checkpoint = load_model_and_config(args.checkpoint, args.config)
    
    # Update config with command line args
    if args.device != 'auto':
        config.training.device = args.device
    if config.training.device == 'auto':
        config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config.evaluation.eval_batch_size = args.batch_size
    config.evaluation.save_predictions = args.save_predictions
    compute_physics = args.physics_analysis and hasattr(model, 'forward_with_derivatives')
    if args.physics_analysis and not compute_physics:
        print("Warning: forward_with_derivatives not implemented; skipping physics analysis")
    config.evaluation.compute_physics_loss = compute_physics
    config.data.data_dir = args.data_dir
    
    # Move model to device
    device = torch.device(config.training.device)
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Setup data module
    print("Setting up data module...")
    data_module = RBDataModule(
        data_dir=config.data.data_dir,
        spatial_downsample=config.data.spatial_downsample,
        temporal_downsample=config.data.temporal_downsample,
        clip_length=config.data.clip_length,
        batch_size=args.batch_size,
        num_workers=4,
        pde_points=config.data.pde_points,
        normalize=config.data.normalize
    )
    
    # Prepare Ra numbers for evaluation
    eval_ra_numbers = config.data.Ra_numbers.copy()
    if args.generalization:
        eval_ra_numbers.extend(args.test_ra)
    
    # Remove duplicates
    eval_ra_numbers = list(set(eval_ra_numbers))
    
    # Setup data for all Ra numbers
    data_module.setup(eval_ra_numbers)
    
    # Setup logger
    experiment_name = f"eval_{config.experiment_name}"
    logger = Logger(
        log_dir=os.path.join(args.output_dir, 'logs'),
        experiment_name=experiment_name,
        use_tensorboard=False,
        use_wandb=False
    )
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = CDAnetEvaluator(
        model=model,
        data_module=data_module,
        config=config.evaluation,
        logger=logger
    )
    
    # Run evaluation on specified splits
    print("Running evaluation...")
    all_results = {}
    
    for Ra in config.data.Ra_numbers:
        for split in args.splits:
            print(f"\nEvaluating Ra={Ra:.0e}, split={split}")
            try:
                metrics = evaluator.evaluate_on_dataset(Ra, split)
                all_results[f"Ra_{Ra:.0e}_{split}"] = metrics
                
                # Print key metrics
                print(f"RRMSE_avg: {metrics.get('RRMSE_avg', 'N/A'):.4f}")
                print(f"MAE_avg: {metrics.get('MAE_avg', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"Failed to evaluate Ra={Ra:.0e}, split={split}: {e}")
    
    # Generalization testing
    if args.generalization:
        print("\nTesting generalization...")
        train_ra = config.data.Ra_numbers[0]
        gen_results = evaluator.evaluate_generalization(train_ra, args.test_ra)
        all_results.update(gen_results)
        
        print("\nGeneralization Results:")
        for ra_key, metrics in gen_results.items():
            if metrics:
                rrmse = metrics.get('RRMSE_avg', float('inf'))
                print(f"{ra_key}: RRMSE_avg = {rrmse:.4f}")
    
    # Temporal evolution analysis
    if args.temporal_evolution:
        print("\nAnalyzing temporal evolution...")
        for Ra in config.data.Ra_numbers:
            print(f"Temporal analysis for Ra={Ra:.0e}")
            temporal_metrics = evaluator.evaluate_temporal_evolution(Ra, 'test')
            
            # Save temporal metrics
            temporal_file = os.path.join(args.output_dir, f'temporal_metrics_Ra_{Ra:.0e}.npz')
            np.savez(temporal_file, **temporal_metrics)
    
    # Create comprehensive evaluation report
    print("\nCreating evaluation report...")
    evaluator.create_evaluation_report(args.output_dir)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("CDAnet Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Evaluation date: {torch.utils.data.dataloader._get_worker_info()}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 30 + "\n")
        for key, metrics in all_results.items():
            if isinstance(metrics, dict):
                f.write(f"\n{key}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric_name:20s}: {value:.6f}\n")
    
    # Print completion message
    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary of key results
    print("\nKey Results:")
    print("-" * 30)
    for key, metrics in all_results.items():
        if isinstance(metrics, dict) and 'RRMSE_avg' in metrics:
            print(f"{key:25s}: RRMSE = {metrics['RRMSE_avg']:.4f}")
    
    print("=" * 60)
    
    # Close logger
    logger.close()


if __name__ == '__main__':
    main()
