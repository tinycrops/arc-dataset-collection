import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from arc_model import JellyfishARCSystem, plot_arc_grid

# Load trained model if available
def load_trained_model(model_path="arc_controller.pth"):
    """Load a trained model from saved weights."""
    # Define the same action map as used during training
    action_map = {
        0: ('rotate_180', lambda grid: np.rot90(grid, k=2)),
        1: ('rotate_90', lambda grid: np.rot90(grid, k=1)),
        2: ('rotate_270', lambda grid: np.rot90(grid, k=3)),
        3: ('move_right', lambda grid: np.roll(grid, shift=1, axis=1)),
        4: ('move_left', lambda grid: np.roll(grid, shift=-1, axis=1)),
        5: ('noop', lambda grid: grid.copy()),
        6: ('flip_horizontal', lambda grid: np.fliplr(grid)),
        7: ('flip_vertical', lambda grid: np.flipud(grid)),
        8: ('move_down', lambda grid: np.roll(grid, shift=1, axis=0)),
        9: ('move_up', lambda grid: np.roll(grid, shift=-1, axis=0)),
        10: ('fill_largest_blue', lambda grid: grid.copy())  # Placeholder, we won't use this
    }
    
    # Create model with the right number of actions
    model = JellyfishARCSystem(symbolic_action_map=action_map)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model.controller.load_state_dict(torch.load(model_path))
    else:
        print("No trained model found. Using initialized model with random weights.")
    
    return model

# Find some ARC tasks to test
def find_arc_tasks(dataset_dir="dataset", num_tasks=5):
    """Find some ARC tasks to test."""
    tasks = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        task_data = json.load(f)
                    
                    if 'train' in task_data and 'test' in task_data and len(tasks) < num_tasks:
                        # Take first training and test examples
                        task = {
                            'train_input': np.array(task_data['train'][0]['input']),
                            'train_output': np.array(task_data['train'][0]['output']),
                            'test_input': np.array(task_data['test'][0]['input']),
                            'test_output': np.array(task_data['test'][0]['output']),
                            'file_path': file_path
                        }
                        tasks.append(task)
                        
                        if len(tasks) >= num_tasks:
                            break
                except Exception as e:
                    print(f"Error loading {os.path.join(root, file)}: {e}")
        
        if len(tasks) >= num_tasks:
            break
    
    return tasks

def main():
    # Load model
    model = load_trained_model()
    
    # Find test tasks
    tasks = find_arc_tasks()
    
    if not tasks:
        print("No tasks found for testing. Using synthetic examples instead.")
        # Create synthetic examples for testing
        tasks = [{
            'train_input': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            'train_output': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'test_input': np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]]),
            'test_output': np.array([[0, 2, 2, 0], [2, 2, 2, 2], [0, 2, 2, 0]]),
            'file_path': "synthetic_example_1"
        }, {
            'train_input': np.array([[1, 2], [3, 4]]),
            'train_output': np.array([[4, 3], [2, 1]]),
            'test_input': np.array([[5, 6], [7, 8]]),
            'test_output': np.array([[8, 7], [6, 5]]),
            'file_path': "synthetic_example_2"
        }]
    
    # Test each task
    correct = 0
    total = len(tasks)
    
    fig, axes = plt.subplots(len(tasks), 4, figsize=(16, 4*len(tasks)))
    
    for i, task in enumerate(tasks):
        # Process both train and test inputs
        train_output_pred, train_action = model.process_grid(task['train_input'])
        test_output_pred, test_action = model.process_grid(task['test_input'])
        
        # Check if test output matches
        is_correct = np.array_equal(test_output_pred, task['test_output'])
        if is_correct:
            correct += 1
        
        # Plot results
        ax_row = axes[i] if len(tasks) > 1 else axes
        
        # Training example
        plot_arc_grid(ax_row[0], task['train_input'], title="Train Input")
        plot_arc_grid(ax_row[1], task['train_output'], title="Expected Train Output")
        
        # Test example
        plot_arc_grid(ax_row[2], task['test_input'], title="Test Input")
        
        # Test prediction
        result_title = f"Predicted ({test_action})"
        if is_correct:
            result_title += " ✓"
        else:
            result_title += " ✗"
        plot_arc_grid(ax_row[3], test_output_pred, title=result_title)
        
        # Print info
        print(f"Task {i+1}: {os.path.basename(task['file_path'])}")
        print(f"  Train action: {train_action}")
        print(f"  Test action: {test_action}")
        print(f"  Correct: {is_correct}")
    
    print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig("test_results.png")
    plt.show()

if __name__ == "__main__":
    main() 