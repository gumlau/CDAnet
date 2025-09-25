"""
Multi-Layer Perceptron (MLP) for physics-informed neural network predictions.
Outputs high-resolution fields and enables automatic differentiation for PDE residuals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedMLP(nn.Module):
    """
    Physics-informed MLP that takes features and coordinates as input.
    
    Args:
        feature_dim: Dimension of input features from U-Net
        coord_dim: Dimension of spatio-temporal coordinates (3 for x, y, t)
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output variables (4 for T, p, u, v)
        activation: Activation function ('softplus', 'relu', 'tanh')
    """
    
    def __init__(self, feature_dim=256, coord_dim=3, hidden_dims=[512, 512, 512, 512], 
                 output_dim=4, activation='softplus'):
        super(PhysicsInformedMLP, self).__init__()
        
        self.feature_dim = feature_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        
        # Choose activation function
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Input dimension: features + coordinates
        input_dim = feature_dim + coord_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # First layer: input is already features + coords
                layers.append(nn.Linear(prev_dim, hidden_dim))
            else:
                # Later layers: concatenate coords again
                layers.append(nn.Linear(prev_dim + coord_dim, hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim + coord_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, features, coords):
        """
        Forward pass through the MLP.
        
        Args:
            features: Feature vectors from U-Net [B, N, feature_dim]
            coords: Spatio-temporal coordinates [B, N, coord_dim]
            
        Returns:
            outputs: Predicted fields [B, N, output_dim] (T, p, u, v)
        """
        # Initial input: concatenate features and coordinates
        x = torch.cat([features, coords], dim=-1)
        
        # Pass through hidden layers with coordinate concatenation
        for i, layer in enumerate(self.layers[:-1]):
            # Concatenate coordinates at each hidden layer
            layer_input = torch.cat([x, coords], dim=-1) if i > 0 else x
            x = self.activation(layer(layer_input))
        
        # Final output layer
        final_input = torch.cat([x, coords], dim=-1)
        outputs = self.layers[-1](final_input)
        
        return outputs
    
    def compute_derivatives(self, features, coords, create_graph=True):
        """
        Compute partial derivatives for PDE residual computation.
        
        Args:
            features: Feature vectors from U-Net [B, N, feature_dim]
            coords: Spatio-temporal coordinates [B, N, coord_dim] (x, y, t)
            create_graph: Whether to create computational graph for higher-order derivatives
            
        Returns:
            outputs: Predicted fields [B, N, 4] (T, p, u, v)
            derivatives: Dictionary of partial derivatives
        """
        coords.requires_grad_(True)
        
        # Forward pass
        outputs = self.forward(features, coords)
        
        # Split outputs into individual fields
        T, p, u, v = outputs[..., 0], outputs[..., 1], outputs[..., 2], outputs[..., 3]
        
        # Split coordinates
        x, y, t = coords[..., 0], coords[..., 1], coords[..., 2]
        
        # Compute first-order derivatives
        derivatives = {}
        
        # Temperature derivatives
        T_grad = torch.autograd.grad(T.sum(), coords, create_graph=create_graph, retain_graph=True)[0]
        derivatives['dT_dx'] = T_grad[..., 0]
        derivatives['dT_dy'] = T_grad[..., 1]
        derivatives['dT_dt'] = T_grad[..., 2]
        
        # Pressure derivatives
        p_grad = torch.autograd.grad(p.sum(), coords, create_graph=create_graph, retain_graph=True)[0]
        derivatives['dp_dx'] = p_grad[..., 0]
        derivatives['dp_dy'] = p_grad[..., 1]
        
        # Velocity derivatives
        u_grad = torch.autograd.grad(u.sum(), coords, create_graph=create_graph, retain_graph=True)[0]
        derivatives['du_dx'] = u_grad[..., 0]
        derivatives['du_dy'] = u_grad[..., 1]
        derivatives['du_dt'] = u_grad[..., 2]
        
        v_grad = torch.autograd.grad(v.sum(), coords, create_graph=create_graph, retain_graph=True)[0]
        derivatives['dv_dx'] = v_grad[..., 0]
        derivatives['dv_dy'] = v_grad[..., 1]
        derivatives['dv_dt'] = v_grad[..., 2]
        
        # Second-order derivatives for diffusion terms
        derivatives['d2T_dx2'] = torch.autograd.grad(derivatives['dT_dx'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 0]
        derivatives['d2T_dy2'] = torch.autograd.grad(derivatives['dT_dy'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 1]
        
        derivatives['d2u_dx2'] = torch.autograd.grad(derivatives['du_dx'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 0]
        derivatives['d2u_dy2'] = torch.autograd.grad(derivatives['du_dy'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 1]
        
        derivatives['d2v_dx2'] = torch.autograd.grad(derivatives['dv_dx'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 0]
        derivatives['d2v_dy2'] = torch.autograd.grad(derivatives['dv_dy'].sum(), coords, 
                                                    create_graph=create_graph, retain_graph=True)[0][..., 1]
        
        return outputs, derivatives