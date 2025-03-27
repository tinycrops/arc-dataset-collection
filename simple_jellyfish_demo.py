#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from enhanced_jellyfish_model import (
    ActionInterpreter,
    plot_arc_grid
)

# Set random seed for reproducibility
np.random.seed(42)

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
    print("=== Simplified Jellyfish Action Interpreter Demo ===")
    
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
    
    # Let's try all actions
    print("Testing all actions...")
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    axs = axs.flatten()
    
    for action_idx in range(len(interpreter.action_functions)):
        result = interpreter.apply_action(sample_grid, action_idx)
        action_name = interpreter.action_functions[action_idx].__name__
        plot_arc_grid(axs[action_idx], result, f"Action {action_idx}: {action_name}")
    
    plt.tight_layout()
    plt.savefig("all_actions.png")
    plt.show()
    
    # Let's manually apply a few transformations in sequence
    print("\nApplying a sequence of transformations...")
    
    # Start with our sample grid
    current_grid = sample_grid.copy()
    
    # Apply rotation
    current_grid = interpreter.apply_action(current_grid, 1)  # rotate_90
    
    # Then apply flip horizontal
    current_grid = interpreter.apply_action(current_grid, 4)  # flip_horizontal
    
    # Then apply fill_holes
    current_grid = interpreter.apply_action(current_grid, 7)  # fill_holes
    
    # Show the result
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plot_arc_grid(plt.gca(), sample_grid, "Original Grid")
    
    plt.subplot(1, 2, 2)
    plot_arc_grid(plt.gca(), current_grid, "After Sequence of Actions")
    
    plt.tight_layout()
    plt.savefig("transformation_sequence.png")
    plt.show()
    
    print("Demo completed!")

if __name__ == "__main__":
    main() 