import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy.ndimage import label, find_objects, binary_dilation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration ---
NUM_COLORS = 10
NUM_BASIC_RHOPALIA = 4
NUM_SPECIALIZED_RHOPALIA = 2

# --- Visualization Utils ---
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

# --- "Specialized Eye Kernel" Functions ---
# These are more advanced feature extractors for the enhanced jellyfish

def eye_object_properties(grid):
    """Extract properties of objects in the grid: size, color, position."""
    # For each non-black color, find objects
    features = []
    
    # Create binary grid (non-zero pixels)
    binary_grid = (grid > 0).astype(int)
    labeled_array, num_features = label(binary_grid)
    
    if num_features == 0:
        # No objects found, return zeros
        return np.zeros(30, dtype=np.float32)  # Fixed size output
    
    # Track object properties (size, position, color)
    largest_obj_size = 0
    largest_obj_color = 0
    largest_obj_pos = [0, 0]  # [row_center, col_center]
    
    total_area = grid.shape[0] * grid.shape[1]
    max_dim = max(grid.shape[0], grid.shape[1])
    
    for i in range(1, num_features + 1):
        # Get object mask
        obj_mask = (labeled_array == i)
        obj_size = np.sum(obj_mask)
        
        # Get most common color in the object
        obj_colors = grid[obj_mask]
        color_counter = Counter(obj_colors)
        obj_color = color_counter.most_common(1)[0][0]
        
        # Get center position
        rows, cols = np.where(obj_mask)
        row_center = np.mean(rows) / grid.shape[0]  # Normalized
        col_center = np.mean(cols) / grid.shape[1]  # Normalized
        
        # Track largest object
        if obj_size > largest_obj_size:
            largest_obj_size = obj_size
            largest_obj_color = obj_color
            largest_obj_pos = [row_center, col_center]
        
        # Add features: normalized size, color, position
        obj_features = [
            obj_size / total_area,    # Size as fraction of grid
            obj_color / NUM_COLORS,   # Normalized color
            row_center,               # Normalized row position
            col_center                # Normalized column position
        ]
        features.extend(obj_features)
    
    # Ensure fixed size output by padding/truncating
    target_length = 30  # Arbitrary fixed size
    
    if len(features) > target_length:
        features = features[:target_length]
    elif len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    
    return np.array(features, dtype=np.float32)

def eye_pattern_detector(grid):
    """Detect common patterns: lines, rectangles, filled areas."""
    h, w = grid.shape
    features = []
    
    # 1. Check for horizontal lines
    horizontal_line_count = 0
    for r in range(h):
        # For each color, check if the entire row has that color
        for c in range(1, NUM_COLORS):
            if np.all(grid[r, :] == c):
                horizontal_line_count += 1
                break
    
    # 2. Check for vertical lines
    vertical_line_count = 0
    for c in range(w):
        # For each color, check if the entire column has that color
        for color in range(1, NUM_COLORS):
            if np.all(grid[:, c] == color):
                vertical_line_count += 1
                break
    
    # 3. Detect rectangles - for each color, count connected components that are rectangular
    rect_count = 0
    for color in range(1, NUM_COLORS):
        color_mask = (grid == color).astype(int)
        labeled_array, num_features = label(color_mask)
        
        for i in range(1, num_features + 1):
            obj_mask = (labeled_array == i)
            rows, cols = np.where(obj_mask)
            
            if len(rows) > 0 and len(cols) > 0:
                # Check if it forms a rectangle by comparing area
                min_row, max_row = np.min(rows), np.max(rows)
                min_col, max_col = np.min(cols), np.max(cols)
                rect_area = (max_row - min_row + 1) * (max_col - min_col + 1)
                actual_area = np.sum(obj_mask)
                
                if rect_area == actual_area:
                    rect_count += 1
    
    # 4. Check symmetry
    h_symmetry = 0
    for color in range(1, NUM_COLORS):
        color_mask = (grid == color).astype(int)
        # Check horizontal symmetry (left-right)
        flipped = np.fliplr(color_mask)
        h_symmetry += np.sum(color_mask == flipped) / (h * w)
    
    v_symmetry = 0
    for color in range(1, NUM_COLORS):
        color_mask = (grid == color).astype(int)
        # Check vertical symmetry (up-down)
        flipped = np.flipud(color_mask)
        v_symmetry += np.sum(color_mask == flipped) / (h * w)
    
    # Normalize all features
    max_lines = max(h, w)
    horizontal_line_count /= max_lines
    vertical_line_count /= max_lines
    rect_count /= max(1, (h * w) // 4)  # Normalize by quarter of grid area
    h_symmetry /= NUM_COLORS
    v_symmetry /= NUM_COLORS
    
    features = [
        horizontal_line_count,
        vertical_line_count,
        rect_count,
        h_symmetry,
        v_symmetry
    ]
    
    return np.array(features, dtype=np.float32)

def eye_color_relationships(grid):
    """Detect relationships between colors: adjacency, containment."""
    features = []
    
    # Create adjacency matrix between colors
    adjacency_matrix = np.zeros((NUM_COLORS, NUM_COLORS), dtype=np.float32)
    
    # Detect adjacency by dilating each color and checking for overlap
    for color1 in range(NUM_COLORS):
        mask1 = (grid == color1).astype(int)
        dilated1 = binary_dilation(mask1)
        
        for color2 in range(color1 + 1, NUM_COLORS):
            mask2 = (grid == color2).astype(int)
            
            # Check adjacency: if dilated version of color1 overlaps with color2
            adjacency = np.sum(dilated1 & mask2)
            
            if adjacency > 0:
                # Normalize by total perimeter
                total_perimeter = np.sum(mask1) + np.sum(mask2)
                if total_perimeter > 0:
                    adjacency_matrix[color1, color2] = adjacency / total_perimeter
                    adjacency_matrix[color2, color1] = adjacency / total_perimeter
    
    # Flatten the upper triangular part of the matrix
    for i in range(NUM_COLORS):
        for j in range(i + 1, NUM_COLORS):
            features.append(adjacency_matrix[i, j])
    
    # Check containment: whether one color is surrounded by another
    containment_matrix = np.zeros((NUM_COLORS, NUM_COLORS), dtype=np.float32)
    
    for containing_color in range(NUM_COLORS):
        mask_container = (grid == containing_color).astype(int)
        # Check if this color forms closed shapes that might contain others
        if np.sum(mask_container) > 4:  # Minimum size to potentially contain something
            for contained_color in range(NUM_COLORS):
                if containing_color != contained_color:
                    mask_contained = (grid == contained_color).astype(int)
                    
                    # Find connected components of the contained color
                    labeled_contained, num_contained = label(mask_contained)
                    
                    contained_count = 0
                    total_contained = 0
                    
                    # For each connected component, check if it's surrounded by the container
                    for i in range(1, num_contained + 1):
                        component = (labeled_contained == i)
                        total_contained += 1
                        
                        # Dilate component and check if it overlaps with container
                        dilated = binary_dilation(component)
                        boundary = dilated & ~component
                        
                        # If all boundary pixels are the container color, it's contained
                        if np.all(grid[boundary] == containing_color):
                            contained_count += 1
                    
                    # Normalize by total contained components
                    if total_contained > 0:
                        containment_matrix[containing_color, contained_color] = contained_count / total_contained
    
    # Flatten the containment matrix (excluding diagonal)
    for i in range(NUM_COLORS):
        for j in range(NUM_COLORS):
            if i != j:
                features.append(containment_matrix[i, j])
    
    return np.array(features, dtype=np.float32)

def eye_spatial_layout(grid):
    """Analyze spatial layout: quadrant distribution, center of mass."""
    h, w = grid.shape
    features = []
    
    # Create quadrant masks
    q1_mask = np.zeros_like(grid, dtype=bool)
    q1_mask[:h//2, :w//2] = True
    
    q2_mask = np.zeros_like(grid, dtype=bool)
    q2_mask[:h//2, w//2:] = True
    
    q3_mask = np.zeros_like(grid, dtype=bool)
    q3_mask[h//2:, :w//2] = True
    
    q4_mask = np.zeros_like(grid, dtype=bool)
    q4_mask[h//2:, w//2:] = True
    
    # Calculate color distribution in each quadrant
    for color in range(NUM_COLORS):
        color_mask = (grid == color)
        if np.sum(color_mask) > 0:
            q1_ratio = np.sum(color_mask & q1_mask) / np.sum(color_mask)
            q2_ratio = np.sum(color_mask & q2_mask) / np.sum(color_mask)
            q3_ratio = np.sum(color_mask & q3_mask) / np.sum(color_mask)
            q4_ratio = np.sum(color_mask & q4_mask) / np.sum(color_mask)
            
            features.extend([q1_ratio, q2_ratio, q3_ratio, q4_ratio])
        else:
            features.extend([0, 0, 0, 0])
    
    # Calculate overall center of mass
    rows, cols = np.indices((h, w))
    non_zero_mask = (grid > 0)
    if np.sum(non_zero_mask) > 0:
        center_row = np.sum(rows * non_zero_mask) / np.sum(non_zero_mask) / h
        center_col = np.sum(cols * non_zero_mask) / np.sum(non_zero_mask) / w
        features.extend([center_row, center_col])
    else:
        features.extend([0.5, 0.5])  # Default to center if no non-zero elements
    
    return np.array(features, dtype=np.float32)

# --- Primitive "Eye Kernel" Functions ---
# Basic feature extractors from the original model

def eye_global_grid_size(grid):
    """Extracts global grid dimensions."""
    max_dim = 30.0  # Assume max 30x30
    return np.array([grid.shape[0] / max_dim, grid.shape[1] / max_dim], dtype=np.float32)

def eye_global_color_counts(grid):
    """Extracts color distribution."""
    counts = Counter(grid.flatten())
    total_pixels = grid.shape[0] * grid.shape[1]
    feature = np.array([counts.get(i, 0) / total_pixels for i in range(NUM_COLORS)], dtype=np.float32)
    return feature

def eye_local_neighbor_similarity(grid):
    """Extracts local structural patterns."""
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
    """Counts connected components for each color."""
    max_objects_norm = 10.0  # Normalize by expected max objects
    object_counts = []
    for color in range(1, NUM_COLORS):  # Exclude black background
        binary_grid = (grid == color).astype(int)
        labeled_array, num_features = label(binary_grid)
        object_counts.append(num_features / max_objects_norm)
    return np.array(object_counts, dtype=np.float32)

# --- Enhanced Rhopalium (Sensory Structure) ---
class EnhancedRhopalium:
    """Enhanced sensory structure with specialized and basic 'eyes'."""
    
    def __init__(self, basic_eyes, specialized_eyes=None, has_memory=False):
        """
        Initialize an enhanced Rhopalium.
        
        Args:
            basic_eyes (list): List of basic eye kernel functions.
            specialized_eyes (list): List of specialized eye kernel functions.
            has_memory (bool): Whether this Rhopalium has memory capability.
        """
        self.basic_eyes = basic_eyes
        self.specialized_eyes = specialized_eyes or []
        self.has_memory = has_memory
        
        # Memory for tracking changes if this Rhopalium has memory
        self.previous_features = None
        
        # Feature dimension is calculated lazily
        self._feature_dim = None
    
    def process(self, grid):
        """
        Process the grid through all eyes and combine features.
        
        Args:
            grid (np.ndarray): Input ARC grid.
            
        Returns:
            np.ndarray: Combined feature vector.
        """
        current_features = []
        
        # Process through basic eyes
        for eye_func in self.basic_eyes:
            feature = eye_func(grid)
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            current_features.append(feature.flatten())
        
        # Process through specialized eyes (if any)
        for eye_func in self.specialized_eyes:
            feature = eye_func(grid)
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            current_features.append(feature.flatten())
        
        # Combine all current features
        combined_features = np.concatenate(current_features)
        
        # If this Rhopalium has memory, add change detection features
        if self.has_memory and self.previous_features is not None:
            # Calculate change between current and previous features
            change_features = combined_features - self.previous_features
            
            # Update memory
            self.previous_features = combined_features.copy()
            
            # Concatenate current features and change features
            return np.concatenate([combined_features, change_features])
        else:
            # Update memory if needed
            if self.has_memory:
                self.previous_features = combined_features.copy()
            
            return combined_features
    
    def reset_memory(self):
        """Reset the Rhopalium's memory."""
        self.previous_features = None
    
    @property
    def feature_dim(self):
        """Get the feature dimension produced by this Rhopalium."""
        if self._feature_dim is None:
            # Calculate dimension by processing a dummy grid
            dummy_grid = np.zeros((5, 5), dtype=int)
            features = self.process(dummy_grid)
            self._feature_dim = features.shape[0]
        
        return self._feature_dim 

# --- Main Enhanced Jellyfish Model ---
class EnhancedJellyfishModel(nn.Module):
    """
    A neurosymbolic model inspired by jellyfish nervous system, 
    enhanced with better sensing, reasoning, and acting capabilities.
    """
    
    def __init__(self, 
                 num_actions=10,
                 hidden_dim=128,
                 num_basic_rhopalia=NUM_BASIC_RHOPALIA,
                 num_specialized_rhopalia=NUM_SPECIALIZED_RHOPALIA):
        """
        Initialize the enhanced jellyfish model.
        
        Args:
            num_actions (int): Number of possible actions.
            hidden_dim (int): Size of hidden layers.
            num_basic_rhopalia (int): Number of basic sensing units.
            num_specialized_rhopalia (int): Number of specialized sensing units.
        """
        super().__init__()
        
        # Create sensing rhopalia (both basic and specialized)
        self.basic_rhopalia = []
        self.specialized_rhopalia = []
        
        # Basic rhopalia for fundamental feature extraction
        basic_eye_functions = [
            eye_global_grid_size,
            eye_global_color_counts,
            eye_local_neighbor_similarity,
            eye_object_count
        ]
        
        for i in range(num_basic_rhopalia):
            # Distribute eye functions to basic rhopalia
            eye_index = i % len(basic_eye_functions)
            self.basic_rhopalia.append(
                EnhancedRhopalium(
                    basic_eyes=[basic_eye_functions[eye_index]],
                    has_memory=(i % 2 == 0)  # Give memory to half of them
                )
            )
        
        # Specialized rhopalia for advanced feature extraction
        specialized_eye_functions = [
            eye_object_properties,
            eye_pattern_detector,
            eye_color_relationships,
            eye_spatial_layout
        ]
        
        for i in range(num_specialized_rhopalia):
            # Distribute eye functions to specialized rhopalia
            eye_index = i % len(specialized_eye_functions)
            self.specialized_rhopalia.append(
                EnhancedRhopalium(
                    basic_eyes=[],  # No basic eyes
                    specialized_eyes=[specialized_eye_functions[eye_index]],
                    has_memory=True  # All specialized rhopalia have memory
                )
            )
        
        # Calculate total feature dimension
        dummy_grid = np.zeros((5, 5), dtype=int)
        features_dim = 0
        
        for rhopalium in self.basic_rhopalia + self.specialized_rhopalia:
            features = rhopalium.process(dummy_grid)
            features_dim += features.shape[0]
        
        # Create neural processing - simplified architecture
        self.feature_layer1 = nn.Linear(features_dim, hidden_dim)
        self.feature_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Final action selection layers
        self.action_layer1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.action_layer2 = nn.Linear(hidden_dim // 4, num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Process input grid to predict the most appropriate action.
        
        Args:
            x (torch.Tensor): Batch of input grids [batch_size, height, width]
            
        Returns:
            torch.Tensor: Action probabilities [batch_size, num_actions]
        """
        batch_size = x.shape[0]
        
        # Process the batch through the rhopalia (sensing) layer
        all_features = []
        
        for i in range(batch_size):
            # Convert tensor to numpy for feature extraction
            grid = x[i].detach().cpu().numpy()
            
            # Extract features from all rhopalia
            instance_features = []
            
            # Basic rhopalia
            for rhopalium in self.basic_rhopalia:
                features = rhopalium.process(grid)
                instance_features.append(torch.tensor(features, dtype=torch.float32))
            
            # Specialized rhopalia
            for rhopalium in self.specialized_rhopalia:
                features = rhopalium.process(grid)
                instance_features.append(torch.tensor(features, dtype=torch.float32))
            
            # Concatenate all features for this instance
            instance_features = torch.cat(instance_features)
            all_features.append(instance_features)
        
        # Stack all instances' features
        features = torch.stack(all_features).to(x.device)
        
        # Process through neural layers - simplified forward pass
        x = F.relu(self.feature_layer1(features))
        x = self.dropout(x)
        x = F.relu(self.feature_layer2(x))
        
        # No attention mechanism for simplicity
        # Final action selection
        x = F.relu(self.action_layer1(x))
        x = self.dropout(x)
        x = self.action_layer2(x)
        
        return F.log_softmax(x, dim=1)
    
    def predict(self, grid):
        """
        Predict action for a single grid.
        
        Args:
            grid (np.ndarray): Input ARC grid.
            
        Returns:
            int: Predicted action index.
        """
        # Convert to tensor and add batch dimension
        grid_tensor = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
        
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            output = self.forward(grid_tensor)
            _, predicted = torch.max(output, 1)
        
        return predicted.item()
    
    def reset_memory(self):
        """Reset memory in all rhopalia."""
        for rhopalium in self.basic_rhopalia + self.specialized_rhopalia:
            rhopalium.reset_memory()

# --- Action Interpreter ---
class ActionInterpreter:
    """
    Interprets the action predicted by the jellyfish model and applies it to the input grid.
    """
    def __init__(self):
        self.action_functions = {
            0: self.identity,
            1: self.rotate_90,
            2: self.rotate_180,
            3: self.rotate_270,
            4: self.flip_horizontal,
            5: self.flip_vertical,
            6: self.transpose,
            7: self.fill_holes,
            8: self.extract_objects,
            9: self.color_transform
        }
    
    def apply_action(self, grid, action_idx):
        """Apply the action to the input grid."""
        if action_idx in self.action_functions:
            return self.action_functions[action_idx](grid)
        return grid  # Default: return unchanged grid
    
    def identity(self, grid):
        """Return the grid unchanged."""
        return grid.copy()
    
    def rotate_90(self, grid):
        """Rotate the grid 90 degrees clockwise."""
        return np.rot90(grid, k=1, axes=(1, 0))  # Clockwise rotation
    
    def rotate_180(self, grid):
        """Rotate the grid 180 degrees."""
        return np.rot90(grid, k=2)
    
    def rotate_270(self, grid):
        """Rotate the grid 270 degrees clockwise (90 counterclockwise)."""
        return np.rot90(grid, k=1)
    
    def flip_horizontal(self, grid):
        """Flip the grid horizontally."""
        return np.fliplr(grid)
    
    def flip_vertical(self, grid):
        """Flip the grid vertically."""
        return np.flipud(grid)
    
    def transpose(self, grid):
        """Transpose the grid."""
        return grid.T
    
    def fill_holes(self, grid):
        """Fill holes inside objects."""
        result = grid.copy()
        
        # For each color, fill holes
        for color in range(1, NUM_COLORS):  # Skip black background
            color_mask = (grid == color).astype(np.uint8)
            
            # Skip if no pixels of this color
            if not np.any(color_mask):
                continue
            
            # Find holes (connected components of zeros surrounded by the color)
            # Create inverted mask (1 for background, 0 for color)
            inv_mask = 1 - color_mask
            
            # Label connected components in the inverted mask
            labeled_holes, num_holes = label(inv_mask)
            
            # For each component in the inverted mask
            for i in range(1, num_holes + 1):
                hole_mask = (labeled_holes == i)
                
                # Check if it's fully enclosed by checking its boundary
                dilated = binary_dilation(hole_mask)
                boundary = dilated & ~hole_mask
                
                # If all boundary pixels are the current color, it's a hole
                if np.all(grid[boundary] == color):
                    # Fill the hole with the color
                    result[hole_mask] = color
        
        return result
    
    def extract_objects(self, grid):
        """Extract and preserve only the largest object of each color."""
        result = np.zeros_like(grid)
        
        # For each color, find the largest object
        for color in range(1, NUM_COLORS):  # Skip black background
            color_mask = (grid == color).astype(np.uint8)
            
            # Skip if no pixels of this color
            if not np.any(color_mask):
                continue
            
            # Label connected components
            labeled_array, num_features = label(color_mask)
            
            # Find largest connected component
            largest_size = 0
            largest_label = 0
            
            for i in range(1, num_features + 1):
                size = np.sum(labeled_array == i)
                if size > largest_size:
                    largest_size = size
                    largest_label = i
            
            # Keep only the largest connected component
            if largest_label > 0:
                largest_mask = (labeled_array == largest_label)
                result[largest_mask] = color
        
        return result
    
    def color_transform(self, grid):
        """Transform colors: shift all colors by 1 (modulo NUM_COLORS)."""
        result = np.zeros_like(grid)
        
        # Loop through each color
        for color in range(NUM_COLORS):
            # Find pixels of this color
            mask = (grid == color)
            
            # Transform color (shift by 1, wrap around at NUM_COLORS)
            new_color = (color + 1) % NUM_COLORS
            
            # Apply transformed color
            result[mask] = new_color
        
        return result

# --- Training and Evaluation Utilities ---
def train_enhanced_jellyfish(model, data_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the enhanced jellyfish model.
    
    Args:
        model (EnhancedJellyfishModel): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate.
        
    Returns:
        list: Training losses per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (grids, actions) in enumerate(data_loader):
            grids = grids.to(device)
            actions = actions.to(device)
            
            # Reset model memory at the start of each batch
            model.reset_memory()
            
            # Forward pass
            outputs = model(grids)
            
            # Calculate loss
            loss = criterion(outputs, actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    return losses

def evaluate_enhanced_jellyfish(model, data_loader):
    """
    Evaluate the enhanced jellyfish model.
    
    Args:
        model (EnhancedJellyfishModel): The model to evaluate.
        data_loader (DataLoader): DataLoader for evaluation data.
        
    Returns:
        float: Accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for grids, actions in data_loader:
            grids = grids.to(device)
            actions = actions.to(device)
            
            # Reset model memory
            model.reset_memory()
            
            # Forward pass
            outputs = model(grids)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
    
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    
    return accuracy

# --- Visualization Function for Debugging ---
def visualize_action_results(model, input_grid, interpreter=None):
    """
    Visualize the results of applying different actions to the input grid.
    
    Args:
        model (EnhancedJellyfishModel): The trained model.
        input_grid (np.ndarray): Input ARC grid.
        interpreter (ActionInterpreter, optional): Action interpreter.
    """
    if interpreter is None:
        interpreter = ActionInterpreter()
    
    # Get model prediction
    predicted_action = model.predict(input_grid)
    
    # Generate results for all possible actions
    fig, axs = plt.subplots(3, 4, figsize=(15, 12))
    axs = axs.flatten()
    
    # Plot input grid
    plot_arc_grid(axs[0], input_grid, "Input Grid")
    
    # Plot grid after applying predicted action
    predicted_result = interpreter.apply_action(input_grid, predicted_action)
    plot_arc_grid(axs[1], predicted_result, f"Predicted: Action {predicted_action}")
    
    # Plot results of all possible actions
    for action_idx in range(len(interpreter.action_functions)):
        if action_idx != predicted_action:  # Skip the predicted action (already shown)
            result = interpreter.apply_action(input_grid, action_idx)
            ax_idx = action_idx + 2 if action_idx < predicted_action else action_idx + 1
            plot_arc_grid(axs[ax_idx], result, f"Action {action_idx}")
    
    plt.tight_layout()
    plt.show() 