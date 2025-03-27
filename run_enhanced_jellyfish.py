#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from enhanced_jellyfish_model import (
    EnhancedJellyfishModel, 
    ActionInterpreter, 
    visualize_action_results,
    plot_arc_grid
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_sample_grid(size=(10, 10)):
    """Create a sample ARC grid for testing."""
    grid = np.zeros(size, dtype=int)
    
    # Add a red rectangle
    grid[2:5, 3:7] = 2
    
    # Add a blue circle-like shape
    grid[6, 4:7] = 1
    grid[7, 3:8] = 1
    grid[8, 4:7] = 1
    
    # Add some yellow pixels
    grid[1, 1] = 4
    grid[1, 8] = 4
    grid[8, 1] = 4
    grid[8, 8] = 4
    
    return grid

def main():
    print("=== Enhanced Jellyfish Model Demo ===")
    
    # Create model
    print("Creating enhanced jellyfish model...")
    model = EnhancedJellyfishModel(
        num_actions=10,
        hidden_dim=128,
        num_basic_rhopalia=4,
        num_specialized_rhopalia=2
    )
    
    # Create action interpreter
    interpreter = ActionInterpreter()
    
    # Create a sample grid
    print("Creating sample grid...")
    sample_grid = create_sample_grid()
    
    # Plot the sample grid
    plt.figure(figsize=(6, 6))
    plot_arc_grid(plt.gca(), sample_grid, "Sample Input Grid")
    plt.tight_layout()
    plt.savefig("sample_grid.png")
    plt.close()
    
    print(f"Sample grid shape: {sample_grid.shape}")
    
    # Reset model memory
    model.reset_memory()
    
    # Get action prediction
    print("Predicting action...")
    predicted_action = model.predict(sample_grid)
    print(f"Predicted action: {predicted_action}")
    
    # Apply the predicted action
    transformed_grid = interpreter.apply_action(sample_grid, predicted_action)
    
    # Create results visualization
    print("Visualizing results...")
    visualize_action_results(model, sample_grid, interpreter)
    
    print("Testing all actions...")
    # Test all actions
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    axs = axs.flatten()
    
    for action_idx, action_name in enumerate(interpreter.action_functions.keys()):
        result = interpreter.apply_action(sample_grid, action_idx)
        action_name = interpreter.action_functions[action_idx].__name__
        plot_arc_grid(axs[action_idx], result, f"Action {action_idx}: {action_name}")
    
    plt.tight_layout()
    plt.savefig("all_actions.png")
    plt.show()
    
    print("Demo completed!")

if __name__ == "__main__":
    main() 