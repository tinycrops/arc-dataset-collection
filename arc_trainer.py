import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from arc_model import JellyfishARCSystem, plot_arc_grid, NUM_COLORS

# --- Configuration ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 5  # Reduced from 50 for testing
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation

class ARCDataset(Dataset):
    """Dataset class for loading ARC tasks."""
    
    def __init__(self, dataset_dirs):
        """
        Initialize the ARC dataset.
        
        Args:
            dataset_dirs: List of directories containing ARC task files
        """
        self.tasks = []
        # Load all tasks from the specified directories
        for dataset_dir in dataset_dirs:
            self._load_tasks_from_dir(dataset_dir)
        
        print(f"Loaded {len(self.tasks)} tasks in total")
    
    def _load_tasks_from_dir(self, base_dir):
        """
        Recursively load all JSON files from a directory.
        
        Args:
            base_dir: The base directory to search for task files
        """
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            task_data = json.load(f)
                            
                        # Process train and test examples
                        if 'train' in task_data and 'test' in task_data:
                            # Add each train/test pair as a separate task
                            for train_example in task_data['train']:
                                for test_example in task_data['test']:
                                    self.tasks.append({
                                        'input': train_example['input'],
                                        'output': train_example['output'],
                                        'test_input': test_example['input'],
                                        'test_output': test_example['output'],
                                        'file_path': file_path
                                    })
                    except Exception as e:
                        print(f"Error loading {os.path.join(root, file)}: {e}")
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Convert to numpy arrays with proper dimensions
        input_grid = np.array(task['input'], dtype=np.int32)
        output_grid = np.array(task['output'], dtype=np.int32)
        test_input_grid = np.array(task['test_input'], dtype=np.int32)
        test_output_grid = np.array(task['test_output'], dtype=np.int32)
        
        return {
            'input': input_grid,
            'output': output_grid,
            'test_input': test_input_grid,
            'test_output': test_output_grid,
            'file_path': task['file_path']
        }


class ARCTrainer:
    """Trainer class for the JellyfishARCSystem."""
    
    def __init__(self, model, actions, device='cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The JellyfishARCSystem model to train
            actions: List of available actions
            device: The device to use for training ('cpu' or 'cuda')
        """
        self.model = model
        self.actions = actions
        self.device = device
        
        # Action mapping for training (name to index)
        self.action_to_idx = {action[0]: idx for idx, action in model.symbolic_action_map.items()}
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.controller.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
    
    def _infer_required_action(self, input_grid, output_grid):
        """
        Infer which action transforms input_grid to output_grid.
        
        Args:
            input_grid: The input grid
            output_grid: The target output grid
            
        Returns:
            action_idx: Index of the action that transforms input_grid to output_grid
            or None if no matching action is found
        """
        # Try each action and see if it produces the target output
        for action_idx, (action_name, action_func) in self.model.symbolic_action_map.items():
            transformed_grid = action_func(input_grid)
            if np.array_equal(transformed_grid, output_grid):
                return action_idx
        
        # If no matching action is found
        return None
    
    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader containing training examples
            
        Returns:
            avg_loss: Average loss for the epoch
            accuracy: Training accuracy
        """
        self.model.controller.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, batch in enumerate(dataloader):
            inputs = batch['input']
            outputs = batch['output']
            
            batch_size = len(inputs)
            action_labels = []
            
            # For each example, determine the correct action
            valid_examples = []
            valid_labels = []
            
            for j in range(batch_size):
                # Convert tensors to numpy arrays
                if isinstance(inputs[j], torch.Tensor):
                    input_grid = inputs[j].numpy()
                else:
                    input_grid = inputs[j]
                    
                if isinstance(outputs[j], torch.Tensor):
                    output_grid = outputs[j].numpy()
                else:
                    output_grid = outputs[j]
                
                # Find which action produces this transformation
                action_idx = self._infer_required_action(input_grid, output_grid)
                
                if action_idx is not None:
                    valid_examples.append(j)
                    valid_labels.append(action_idx)
            
            # Skip if no valid examples in this batch
            if not valid_examples:
                continue
            
            # Extract valid examples
            valid_inputs = [inputs[j].numpy() if isinstance(inputs[j], torch.Tensor) else inputs[j] 
                           for j in valid_examples]
            
            # Create tensor of labels
            action_labels = torch.tensor(valid_labels, dtype=torch.long).to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass through model for each valid input
            logits_list = []
            for input_grid in valid_inputs:
                # Process through rhopalia
                all_features = [rhop.process(input_grid) for rhop in self.model.rhopalia]
                controller_input = np.concatenate(all_features)
                controller_input_tensor = torch.from_numpy(controller_input).float().unsqueeze(0).to(self.device)
                
                # Get action logits
                logits = self.model.controller(controller_input_tensor)
                logits_list.append(logits)
            
            # Combine all logits
            if logits_list:
                all_logits = torch.cat(logits_list, dim=0)
                
                # Compute loss
                loss = self.criterion(all_logits, action_labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item() * len(valid_examples)
                _, predicted = torch.max(all_logits, 1)
                correct += (predicted == action_labels).sum().item()
                total += len(valid_examples)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f'Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader containing validation examples
            
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.controller.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input']
                outputs = batch['output']
                
                batch_size = len(inputs)
                valid_examples = []
                valid_labels = []
                
                for j in range(batch_size):
                    # Convert tensors to numpy arrays
                    if isinstance(inputs[j], torch.Tensor):
                        input_grid = inputs[j].numpy()
                    else:
                        input_grid = inputs[j]
                        
                    if isinstance(outputs[j], torch.Tensor):
                        output_grid = outputs[j].numpy()
                    else:
                        output_grid = outputs[j]
                    
                    # Find which action produces this transformation
                    action_idx = self._infer_required_action(input_grid, output_grid)
                    
                    if action_idx is not None:
                        valid_examples.append(j)
                        valid_labels.append(action_idx)
                
                # Skip if no valid examples
                if not valid_examples:
                    continue
                
                # Extract valid examples
                valid_inputs = [inputs[j].numpy() if isinstance(inputs[j], torch.Tensor) else inputs[j] 
                               for j in valid_examples]
                
                # Create tensor of labels
                action_labels = torch.tensor(valid_labels, dtype=torch.long).to(self.device)
                
                # Forward pass for each valid input
                logits_list = []
                for input_grid in valid_inputs:
                    # Process through rhopalia
                    all_features = [rhop.process(input_grid) for rhop in self.model.rhopalia]
                    controller_input = np.concatenate(all_features)
                    controller_input_tensor = torch.from_numpy(controller_input).float().unsqueeze(0).to(self.device)
                    
                    # Get action logits
                    logits = self.model.controller(controller_input_tensor)
                    logits_list.append(logits)
                
                # Combine all logits
                if logits_list:
                    all_logits = torch.cat(logits_list, dim=0)
                    
                    # Compute loss
                    loss = self.criterion(all_logits, action_labels)
                    
                    # Statistics
                    running_loss += loss.item() * len(valid_examples)
                    _, predicted = torch.max(all_logits, 1)
                    correct += (predicted == action_labels).sum().item()
                    total += len(valid_examples)
        
        avg_loss = running_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate_on_test(self, dataloader, num_examples=5):
        """
        Evaluate the model on test examples and visualize results.
        
        Args:
            dataloader: DataLoader containing test examples
            num_examples: Number of examples to visualize
        """
        self.model.controller.eval()
        
        correct = 0
        total = 0
        examples_to_show = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Get only a few examples for visualization
                if len(examples_to_show) >= num_examples:
                    break
                
                test_inputs = batch['test_input']
                test_outputs = batch['test_output']
                
                for j in range(len(test_inputs)):
                    # Convert tensors to numpy arrays
                    if isinstance(test_inputs[j], torch.Tensor):
                        test_input = test_inputs[j].numpy()
                    else:
                        test_input = test_inputs[j]
                        
                    if isinstance(test_outputs[j], torch.Tensor):
                        true_output = test_outputs[j].numpy()
                    else:
                        true_output = test_outputs[j]
                    
                    # Process through model
                    predicted_output, action_name = self.model.process_grid(test_input)
                    
                    # Check if correct
                    is_correct = np.array_equal(predicted_output, true_output)
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Save example for visualization
                    if len(examples_to_show) < num_examples:
                        examples_to_show.append({
                            'input': test_input,
                            'true_output': true_output,
                            'predicted_output': predicted_output,
                            'action': action_name,
                            'correct': is_correct
                        })
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        print(f'Test Accuracy: {accuracy:.4f} ({correct}/{total})')
        
        # Visualize examples
        if examples_to_show:
            fig, axes = plt.subplots(len(examples_to_show), 3, figsize=(12, 4*len(examples_to_show)))
            
            for i, example in enumerate(examples_to_show):
                if len(examples_to_show) == 1:
                    ax_row = axes
                else:
                    ax_row = axes[i]
                
                # Plot input
                plot_arc_grid(ax_row[0], example['input'], title="Input")
                
                # Plot true output
                plot_arc_grid(ax_row[1], example['true_output'], title="True Output")
                
                # Plot predicted output
                result_title = f"Predicted ({example['action']})"
                if example['correct']:
                    result_title += " ✓"
                else:
                    result_title += " ✗"
                plot_arc_grid(ax_row[2], example['predicted_output'], title=result_title)
            
            plt.tight_layout()
            plt.show()
        
        return accuracy


def process_dataset_for_symbolic_actions(dataset_path="dataset"):
    """
    Analyze tasks in the dataset to extract commonly needed operations.
    Returns statistics about which operations are most common.
    """
    print(f"Analyzing dataset at: {dataset_path}")
    
    # Map to store operation frequency
    operation_counts = {}
    
    # Identify some common operations
    def check_is_identity(input_grid, output_grid):
        return np.array_equal(input_grid, output_grid)
    
    def check_is_rotation_90(input_grid, output_grid):
        return np.array_equal(np.rot90(input_grid, k=1), output_grid)
    
    def check_is_rotation_180(input_grid, output_grid):
        return np.array_equal(np.rot90(input_grid, k=2), output_grid)
    
    def check_is_rotation_270(input_grid, output_grid):
        return np.array_equal(np.rot90(input_grid, k=3), output_grid)
    
    def check_is_flip_horizontal(input_grid, output_grid):
        return np.array_equal(np.fliplr(input_grid), output_grid)
    
    def check_is_flip_vertical(input_grid, output_grid):
        return np.array_equal(np.flipud(input_grid), output_grid)
    
    def check_is_move_up(input_grid, output_grid):
        return np.array_equal(np.roll(input_grid, shift=-1, axis=0), output_grid)
    
    def check_is_move_down(input_grid, output_grid):
        return np.array_equal(np.roll(input_grid, shift=1, axis=0), output_grid)
    
    def check_is_move_left(input_grid, output_grid):
        return np.array_equal(np.roll(input_grid, shift=-1, axis=1), output_grid)
    
    def check_is_move_right(input_grid, output_grid):
        return np.array_equal(np.roll(input_grid, shift=1, axis=1), output_grid)
    
    # List of operations to check
    operations = [
        ("identity", check_is_identity),
        ("rotation_90", check_is_rotation_90),
        ("rotation_180", check_is_rotation_180),
        ("rotation_270", check_is_rotation_270),
        ("flip_horizontal", check_is_flip_horizontal),
        ("flip_vertical", check_is_flip_vertical),
        ("move_up", check_is_move_up),
        ("move_down", check_is_move_down),
        ("move_left", check_is_move_left),
        ("move_right", check_is_move_right)
    ]
    
    # Create a dataset instance to load the tasks
    try:
        dataset = ARCDataset([dataset_path])
        
        total_examples = 0
        identified_ops = 0
        
        # Initialize counts
        for op_name, _ in operations:
            operation_counts[op_name] = 0
        
        # Check each task
        for i in range(len(dataset)):
            task = dataset[i]
            input_grid = task['input']
            output_grid = task['output']
            
            total_examples += 1
            op_found = False
            
            # Check each operation
            for op_name, op_func in operations:
                if op_func(input_grid, output_grid):
                    operation_counts[op_name] += 1
                    op_found = True
                    break
            
            if op_found:
                identified_ops += 1
        
        # Print results
        print(f"Analyzed {total_examples} examples")
        print(f"Identified operations for {identified_ops} examples ({identified_ops/total_examples*100:.2f}%)")
        
        # Sort operations by frequency
        sorted_ops = sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\nOperation frequencies:")
        for op_name, count in sorted_ops:
            if count > 0:
                print(f"{op_name}: {count} ({count/total_examples*100:.2f}%)")
        
        # Return most common operations
        return [op_name for op_name, _ in sorted_ops if operation_counts[op_name] > 0]
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return []


def create_action_map(common_operations):
    """Create a symbolic action map based on identified common operations."""
    action_map = {}
    idx = 0
    
    # Map operation names to functions
    for op in common_operations:
        if op == "identity":
            from arc_model import op_noop
            action_map[idx] = ('noop', op_noop)
            idx += 1
        elif op == "rotation_90":
            action_map[idx] = ('rotate_90', lambda grid: np.rot90(grid, k=1))
            idx += 1
        elif op == "rotation_180":
            from arc_model import op_rotate_180
            action_map[idx] = ('rotate_180', op_rotate_180)
            idx += 1
        elif op == "rotation_270":
            action_map[idx] = ('rotate_270', lambda grid: np.rot90(grid, k=3))
            idx += 1
        elif op == "flip_horizontal":
            action_map[idx] = ('flip_horizontal', lambda grid: np.fliplr(grid))
            idx += 1
        elif op == "flip_vertical":
            action_map[idx] = ('flip_vertical', lambda grid: np.flipud(grid))
            idx += 1
        elif op == "move_up":
            from arc_model import op_move_up
            action_map[idx] = ('move_up', op_move_up)
            idx += 1
        elif op == "move_down":
            action_map[idx] = ('move_down', lambda grid: np.roll(grid, shift=1, axis=0))
            idx += 1
        elif op == "move_left":
            action_map[idx] = ('move_left', lambda grid: np.roll(grid, shift=-1, axis=1))
            idx += 1
        elif op == "move_right":
            action_map[idx] = ('move_right', lambda grid: np.roll(grid, shift=1, axis=1))
            idx += 1
    
    # Add the fill_largest_blue action from our original model
    from arc_model import op_fill_largest_blue
    action_map[idx] = ('fill_largest_blue', op_fill_largest_blue)
    
    return action_map


def main():
    # Define paths to dataset directories to use
    dataset_dirs = [
        "dataset/arc-dataset-diva/data/land_ccw"  # Using just one directory for testing
    ]
    
    print("Processing dataset to identify common operations...")
    common_operations = process_dataset_for_symbolic_actions("dataset")
    
    # Use top operations to create our action map
    action_map = create_action_map(common_operations[:10])  # Use top 10 operations
    
    # Print available actions
    print("\nAvailable symbolic actions:")
    for idx, (name, _) in action_map.items():
        print(f"{idx}: {name}")
    
    # Create dataset
    dataset = ARCDataset(dataset_dirs)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_idx = int(np.floor(TRAIN_VAL_SPLIT * dataset_size))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    # Create subset samplers
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    # Initialize model
    device = torch.device("cpu")
    model = JellyfishARCSystem(symbolic_action_map=action_map)
    
    # Initialize trainer
    trainer = ARCTrainer(model, action_map, device=device)
    
    # Train model
    print("\nStarting training...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping if needed
        if epoch > 10 and val_accuracies[-1] < val_accuracies[-2] < val_accuracies[-3]:
            print("Validation accuracy decreasing for 3 epochs, stopping early")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Evaluate on test set
    print("\nEvaluating on test examples...")
    test_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
    test_acc = trainer.evaluate_on_test(test_loader, num_examples=5)
    
    # Save model
    torch.save(model.controller.state_dict(), 'arc_controller.pth')
    print("Model saved to arc_controller.pth")


if __name__ == "__main__":
    main() 