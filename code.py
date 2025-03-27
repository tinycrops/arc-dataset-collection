import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration ---
NUM_COLORS = 10
NUM_RHOPALIA = 4
SYMBOLIC_ACTIONS = ['noop', 'move_up', 'fill_largest_blue', 'rotate_180']

# --- Visualization Utils (from previous code) ---
ARC_CMAP_LIST = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
ARC_CMAP = mcolors.ListedColormap(ARC_CMAP_LIST)
ARC_NORM = mcolors.Normalize(vmin=0, vmax=NUM_COLORS-1)

def plot_arc_grid(ax, grid, title="ARC Grid"):
    """Helper function to plot an ARC grid."""
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

# --- "Eye Kernel" Functions ---
# These are simple feature extractors mimicking different eye types.

def eye_global_grid_size(grid):
    """Mimics 'upper lens' - global context."""
    # Normalize size features (e.g., scale to 0-1 based on max expected size)
    max_dim = 30.0 # Assume max 30x30
    return np.array([grid.shape[0] / max_dim, grid.shape[1] / max_dim], dtype=np.float32)

def eye_global_color_counts(grid):
    """Mimics 'upper lens' - overall composition."""
    counts = Counter(grid.flatten())
    total_pixels = grid.shape[0] * grid.shape[1]
    # Ensure all colors are present, normalize counts
    feature = np.array([counts.get(i, 0) / total_pixels for i in range(NUM_COLORS)], dtype=np.float32)
    return feature

def eye_local_neighbor_similarity(grid):
    """Mimics 'lower lens' - local structure, checks avg similarity."""
    padded = np.pad(grid, 1, mode='edge')
    similarity = 0
    count = 0
    for r in range(1, padded.shape[0] - 1):
        for c in range(1, padded.shape[1] - 1):
            center = padded[r, c]
            # Compare with 4 neighbors
            similarity += (padded[r-1, c] == center)
            similarity += (padded[r+1, c] == center)
            similarity += (padded[r, c-1] == center)
            similarity += (padded[r, c+1] == center)
            count += 4
    avg_similarity = similarity / count if count > 0 else 0
    return np.array([avg_similarity], dtype=np.float32)

def eye_object_count(grid):
    """Mimics 'lower lens' - object detection."""
    # Count connected components for each non-black color
    max_objects_norm = 10.0 # Normalize by expected max objects
    object_counts = []
    for color in range(1, NUM_COLORS): # Exclude black background
        binary_grid = (grid == color).astype(int)
        labeled_array, num_features = label(binary_grid)
        object_counts.append(num_features / max_objects_norm)
    return np.array(object_counts, dtype=np.float32) # Returns 9 features

# --- Rhopalium Module ---
class Rhopalium:
    def __init__(self, eye_kernels):
        """
        Initializes a Rhopalium sensory structure.

        Args:
            eye_kernels (list): A list of 'eye kernel' functions.
        """
        self.eye_kernels = eye_kernels
        self._feature_dim = None # Calculate dimension lazily

    def process(self, grid):
        """
        Processes the grid using all associated eye kernels.

        Args:
            grid (np.ndarray): The input ARC grid.

        Returns:
            np.ndarray: A concatenated feature vector from all eyes.
        """
        features = []
        for eye_func in self.eye_kernels:
            feature = eye_func(grid)
            # Ensure feature is a flat numpy array
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            features.append(feature.flatten())

        concatenated_features = np.concatenate(features)
        # Calculate feature dimension on first run
        if self._feature_dim is None:
             self._feature_dim = concatenated_features.shape[0]
        return concatenated_features

    @property
    def feature_dim(self):
        if self._feature_dim is None:
            # Calculate dimension by processing a dummy grid if needed
            dummy_grid = np.zeros((5,5), dtype=int)
            self.process(dummy_grid)
        return self._feature_dim


# --- Neural Controller (Nerve Ring) ---
class Controller(nn.Module):
    def __init__(self, total_rhopalia_feature_dim, num_actions):
        """
        Initializes the central controller (MLP).

        Args:
            total_rhopalia_feature_dim (int): The combined feature dimension
                                               from all Rhopalia.
            num_actions (int): The number of symbolic actions to choose from.
        """
        super().__init__()
        # Example MLP structure - can be made more complex
        hidden_dim = 128
        self.fc1 = nn.Linear(total_rhopalia_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_actions)
        print(f"Controller initialized with input dim {total_rhopalia_feature_dim}")

    def forward(self, x):
        """
        Processes the combined Rhopalia features to select an action.

        Args:
            x (torch.Tensor): Concatenated feature tensor from all Rhopalia.

        Returns:
            torch.Tensor: Logits over the possible symbolic actions.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# --- Symbolic Operations ---
# Functions that perform the actual ARC transformations.

def op_noop(grid):
    """Action: Do nothing."""
    return grid.copy()

def op_move_up(grid):
    """Action: Roll the grid content upwards."""
    return np.roll(grid, shift=-1, axis=0)

def op_fill_largest_blue(grid):
    """Action: Find largest non-black object, fill it with blue (color 1)."""
    output_grid = grid.copy()
    largest_obj_size = -1
    target_coords = None
    target_color_to_fill = 1 # Blue

    for color in range(1, NUM_COLORS): # Exclude black
        binary_grid = (grid == color).astype(int)
        labeled_array, num_features = label(binary_grid)
        if num_features > 0:
            objects = find_objects(labeled_array)
            for i, slc in enumerate(objects):
                obj_mask = (labeled_array[slc] == (i + 1))
                obj_size = np.sum(obj_mask)
                if obj_size > largest_obj_size:
                    largest_obj_size = obj_size
                    # Get coordinates relative to original grid
                    coords = np.where(labeled_array == (i + 1))
                    target_coords = coords # Store tuple of (row_indices, col_indices)

    if target_coords is not None:
        output_grid[target_coords] = target_color_to_fill

    return output_grid

def op_rotate_180(grid):
    """Action: Rotate the grid 180 degrees."""
    return np.rot90(grid, k=2)


# --- Main Jellyfish ARC System ---
class JellyfishARCSystem:
    def __init__(self, num_rhopalia=NUM_RHOPALIA, symbolic_action_map=None):
        """
        Initializes the complete system.

        Args:
            num_rhopalia (int): Number of sensory modules.
            symbolic_action_map (dict): Mapping from index to (name, function).
                                        If None, uses default.
        """
        self.num_rhopalia = num_rhopalia
        self.rhopalia = self._create_rhopalia()

        total_feature_dim = sum(r.feature_dim for r in self.rhopalia)

        if symbolic_action_map is None:
            self.symbolic_action_map = {
                0: ('noop', op_noop),
                1: ('move_up', op_move_up),
                2: ('fill_largest_blue', op_fill_largest_blue),
                3: ('rotate_180', op_rotate_180),
                # Add more actions here...
            }
        else:
            self.symbolic_action_map = symbolic_action_map

        num_actions = len(self.symbolic_action_map)

        self.controller = Controller(total_feature_dim, num_actions)
        # Move controller to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller.to(self.device)

    def _create_rhopalia(self):
        """Creates the Rhopalium modules with assigned eye kernels."""
        # Example assignment - distribute eyes somewhat evenly or functionally
        all_eyes = [
            eye_global_grid_size, eye_global_color_counts,
            eye_local_neighbor_similarity, eye_object_count
        ]
        rhopalia_list = []
        for i in range(self.num_rhopalia):
            # Simple cyclic assignment for this example
            assigned_eyes = [all_eyes[j % len(all_eyes)] for j in range(i, i + (len(all_eyes)//self.num_rhopalia + 1))]
            # Make sure each Rhopalium gets at least one eye if possible
            if not assigned_eyes and all_eyes:
                assigned_eyes = [all_eyes[i % len(all_eyes)]]
            print(f"Rhopalium {i} assigned eyes: {[f.__name__ for f in assigned_eyes]}")
            rhopalia_list.append(Rhopalium(assigned_eyes))
        return rhopalia_list

    def process_grid(self, input_grid):
        """
        Runs the full forward pass: grid -> features -> controller -> action -> output grid.

        Args:
            input_grid (np.ndarray): The ARC task input grid.

        Returns:
            np.ndarray: The predicted output grid.
            str: The name of the selected action.
        """
        # 1. Process grid through all Rhopalia
        all_features = [rhop.process(input_grid) for rhop in self.rhopalia]

        # 2. Concatenate features for the controller
        controller_input = np.concatenate(all_features)
        controller_input_tensor = torch.from_numpy(controller_input).float().unsqueeze(0).to(self.device) # Add batch dim

        # 3. Get action selection from controller
        self.controller.eval() # Set to evaluation mode
        with torch.no_grad(): # We are not training here
            action_logits = self.controller(controller_input_tensor)
            # Use argmax for deterministic selection (could use sampling during training)
            selected_action_index = torch.argmax(action_logits, dim=1).item()

        # 4. Execute the selected symbolic action
        if selected_action_index in self.symbolic_action_map:
            action_name, action_func = self.symbolic_action_map[selected_action_index]
            print(f"Controller selected action: {action_name} (index {selected_action_index})")
            output_grid = action_func(input_grid)
        else:
            print(f"Warning: Controller selected invalid action index {selected_action_index}. Performing no-op.")
            action_name = 'invalid_noop'
            output_grid = op_noop(input_grid)

        return output_grid, action_name


# --- Example Usage ---

# Create a sample ARC-like grid
sample_grid_input = np.array([
    [0, 0, 0, 0, 0],
    [0, 2, 2, 2, 0],
    [0, 2, 0, 2, 0],
    [0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0]
], dtype=int)

# Initialize the system
print("--- Initializing System ---")
jelly_system = JellyfishARCSystem(symbolic_action_map={
    0: ('noop', op_noop),
    1: ('move_up', op_move_up),
    2: ('fill_largest_blue', op_fill_largest_blue), # Fill largest object (the '2's) with blue (1)
    3: ('rotate_180', op_rotate_180),
})
print("--- System Initialized ---")

# Process the grid
predicted_grid, action_taken = jelly_system.process_grid(sample_grid_input)

# Visualize Input and Output
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plot_arc_grid(axes[0], sample_grid_input, title="Input Grid")
plot_arc_grid(axes[1], predicted_grid, title=f"Predicted Output (Action: {action_taken})")
plt.tight_layout()
plt.show()

# Example 2: Grid likely to trigger move_up if learned
sample_grid_input_2 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [3, 3, 3, 3],
    [0, 0, 0, 0],
], dtype=int)

predicted_grid_2, action_taken_2 = jelly_system.process_grid(sample_grid_input_2)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
plot_arc_grid(axes[0], sample_grid_input_2, title="Input Grid 2")
plot_arc_grid(axes[1], predicted_grid_2, title=f"Predicted Output 2 (Action: {action_taken_2})")
plt.tight_layout()
plt.show()