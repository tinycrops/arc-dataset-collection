import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import matplotlib.colors as mcolors
from scipy.ndimage import label
from tqdm import tqdm

# --- Configuration ---
NUM_COLORS = 10

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

class ARCTaskAnalyzer:
    """Analyzer for ARC tasks to extract patterns, transformations, and statistics."""
    
    def __init__(self):
        """Initialize the analyzer."""
        # Basic operations to check
        self.basic_operations = {
            "identity": lambda in_grid, out_grid: np.array_equal(in_grid, out_grid),
            "rotation_90": lambda in_grid, out_grid: self._check_rotation(in_grid, out_grid, k=1),
            "rotation_180": lambda in_grid, out_grid: self._check_rotation(in_grid, out_grid, k=2),
            "rotation_270": lambda in_grid, out_grid: self._check_rotation(in_grid, out_grid, k=3),
            "flip_horizontal": lambda in_grid, out_grid: np.array_equal(np.fliplr(in_grid), out_grid),
            "flip_vertical": lambda in_grid, out_grid: np.array_equal(np.flipud(in_grid), out_grid),
            "shift_up": lambda in_grid, out_grid: self._check_shift(in_grid, out_grid, axis=0, shift=-1),
            "shift_down": lambda in_grid, out_grid: self._check_shift(in_grid, out_grid, axis=0, shift=1),
            "shift_left": lambda in_grid, out_grid: self._check_shift(in_grid, out_grid, axis=1, shift=-1),
            "shift_right": lambda in_grid, out_grid: self._check_shift(in_grid, out_grid, axis=1, shift=1),
            "color_change": lambda in_grid, out_grid: self._check_color_change(in_grid, out_grid),
            "size_change": lambda in_grid, out_grid: self._check_size_change(in_grid, out_grid),
            "object_counting": lambda in_grid, out_grid: self._check_object_counting(in_grid, out_grid),
            "object_addition": lambda in_grid, out_grid: self._check_object_addition(in_grid, out_grid),
            "object_removal": lambda in_grid, out_grid: self._check_object_removal(in_grid, out_grid),
        }
        
        # Statistics
        self.task_statistics = {
            "grid_sizes": Counter(),
            "color_counts": Counter(),
            "operation_counts": Counter(),
            "object_counts": Counter(),
            "complexity_scores": [],
            "transformation_categories": Counter(),
        }
        
        # Operations identified per task
        self.task_operations = {}
        
    def _check_rotation(self, in_grid, out_grid, k=1):
        """Check if out_grid is a rotation of in_grid by k*90 degrees."""
        return np.array_equal(np.rot90(in_grid, k=k), out_grid)
    
    def _check_shift(self, in_grid, out_grid, axis=0, shift=1):
        """Check if out_grid is in_grid shifted along the specified axis."""
        if in_grid.shape != out_grid.shape:
            return False
        return np.array_equal(np.roll(in_grid, shift=shift, axis=axis), out_grid)
    
    def _check_color_change(self, in_grid, out_grid):
        """Check if out_grid is in_grid with consistent color changes."""
        if in_grid.shape != out_grid.shape:
            return False
        
        # Create a mapping of input colors to output colors
        color_map = {}
        for i in range(in_grid.shape[0]):
            for j in range(in_grid.shape[1]):
                in_color = in_grid[i, j]
                out_color = out_grid[i, j]
                
                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return False
                else:
                    color_map[in_color] = out_color
                    
        # Also check if this is just identity
        if all(k == v for k, v in color_map.items()):
            return False
                
        return True
    
    def _check_size_change(self, in_grid, out_grid):
        """Check if the output is a resized version of the input."""
        # Simple test: is the aspect ratio preserved?
        if in_grid.shape == out_grid.shape:
            return False
            
        in_ratio = in_grid.shape[0] / in_grid.shape[1]
        out_ratio = out_grid.shape[0] / out_grid.shape[1]
        
        return abs(in_ratio - out_ratio) < 0.1  # Allow some tolerance
    
    def _check_object_counting(self, in_grid, out_grid):
        """Check if the output could represent counting objects in the input."""
        # Count objects in input
        in_objects = self._count_objects(in_grid)
        
        # Check if output has a single number (a single connected component)
        out_binary = (out_grid > 0).astype(int)
        labeled_array, num_features = label(out_binary)
        
        # If output has exactly one object, check if its color value could be a count
        if num_features == 1 and out_grid.max() <= in_objects:
            return True
            
        return False
    
    def _check_object_addition(self, in_grid, out_grid):
        """Check if objects were added from input to output."""
        in_objects = self._count_objects(in_grid)
        out_objects = self._count_objects(out_grid)
        
        return out_objects > in_objects and in_grid.shape == out_grid.shape
    
    def _check_object_removal(self, in_grid, out_grid):
        """Check if objects were removed from input to output."""
        in_objects = self._count_objects(in_grid)
        out_objects = self._count_objects(out_grid)
        
        return in_objects > out_objects and in_grid.shape == out_grid.shape
    
    def _count_objects(self, grid):
        """Count the number of objects (connected components) in a grid."""
        # Convert to binary (any non-zero pixel is part of an object)
        binary_grid = (grid > 0).astype(int)
        labeled_array, num_features = label(binary_grid)
        return num_features
    
    def _calculate_complexity_score(self, in_grid, out_grid):
        """Calculate a complexity score for a task pair."""
        # Factors that increase complexity:
        # 1. Grid size
        size_factor = np.prod(in_grid.shape) / 100  # Normalize
        
        # 2. Number of colors
        in_colors = len(np.unique(in_grid))
        out_colors = len(np.unique(out_grid))
        color_factor = (in_colors + out_colors) / 20  # Normalize
        
        # 3. Number of objects
        in_objects = self._count_objects(in_grid)
        out_objects = self._count_objects(out_grid)
        object_factor = (in_objects + out_objects) / 10  # Normalize
        
        # 4. If no simple transformation is identified
        operation_identified = False
        for op_name, op_func in self.basic_operations.items():
            if op_func(in_grid, out_grid):
                operation_identified = True
                break
        operation_factor = 0 if operation_identified else 2
        
        # Combine factors
        complexity = size_factor + color_factor + object_factor + operation_factor
        return complexity
    
    def _categorize_transformation(self, in_grid, out_grid, operations):
        """Categorize the transformation based on identified operations."""
        if len(operations) == 0:
            return "unknown"
            
        categories = {
            "geometric": ["rotation_90", "rotation_180", "rotation_270", "flip_horizontal", "flip_vertical"],
            "shift": ["shift_up", "shift_down", "shift_left", "shift_right"],
            "color": ["color_change"],
            "object": ["object_addition", "object_removal", "object_counting"],
            "size": ["size_change"],
            "identity": ["identity"]
        }
        
        for category, ops in categories.items():
            if any(op in operations for op in ops):
                return category
                
        return "composite"
    
    def analyze_task(self, task_file):
        """Analyze a single ARC task file."""
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_id = os.path.basename(task_file).split('.')[0]
            self.task_operations[task_id] = []
            
            # Process each train example
            for train_example in task_data['train']:
                in_grid = np.array(train_example['input'])
                out_grid = np.array(train_example['output'])
                
                # Record grid sizes
                self.task_statistics['grid_sizes'][str(in_grid.shape)] += 1
                self.task_statistics['grid_sizes'][str(out_grid.shape)] += 1
                
                # Record color counts
                for color in range(NUM_COLORS):
                    if color in in_grid:
                        self.task_statistics['color_counts'][color] += 1
                
                # Record object counts
                in_objects = self._count_objects(in_grid)
                out_objects = self._count_objects(out_grid)
                self.task_statistics['object_counts'][in_objects] += 1
                self.task_statistics['object_counts'][out_objects] += 1
                
                # Identify operations
                example_operations = []
                for op_name, op_func in self.basic_operations.items():
                    if op_func(in_grid, out_grid):
                        example_operations.append(op_name)
                        self.task_statistics['operation_counts'][op_name] += 1
                
                # Record transformation category
                category = self._categorize_transformation(in_grid, out_grid, example_operations)
                self.task_statistics['transformation_categories'][category] += 1
                
                # Calculate complexity score
                complexity = self._calculate_complexity_score(in_grid, out_grid)
                self.task_statistics['complexity_scores'].append(complexity)
                
                self.task_operations[task_id].extend(example_operations)
                
            return {
                'task_id': task_id,
                'operations': list(set(self.task_operations[task_id])),
                'train_examples': len(task_data['train']),
                'test_examples': len(task_data['test']),
                'complexity': np.mean([self._calculate_complexity_score(
                    np.array(ex['input']), 
                    np.array(ex['output'])) 
                    for ex in task_data['train']])
            }
                
        except Exception as e:
            print(f"Error analyzing task {task_file}: {e}")
            return None
    
    def analyze_dataset(self, dataset_dirs, max_tasks=None):
        """Analyze a dataset of ARC tasks."""
        all_files = []
        for dataset_dir in dataset_dirs:
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.json'):
                        all_files.append(os.path.join(root, file))
                        
        if max_tasks and len(all_files) > max_tasks:
            all_files = all_files[:max_tasks]
            
        print(f"Analyzing {len(all_files)} ARC tasks...")
        task_summaries = []
        
        for file in tqdm(all_files):
            summary = self.analyze_task(file)
            if summary:
                task_summaries.append(summary)
                
        return task_summaries
    
    def print_statistics(self):
        """Print statistics about the analyzed tasks."""
        print("\n=== ARC Dataset Statistics ===\n")
        
        print("Grid Sizes (Top 5):")
        for size, count in self.task_statistics['grid_sizes'].most_common(5):
            print(f"  {size}: {count}")
            
        print("\nColor Usage:")
        for color, count in sorted(self.task_statistics['color_counts'].items()):
            print(f"  Color {color}: {count}")
            
        print("\nOperation Frequency:")
        for op, count in self.task_statistics['operation_counts'].most_common():
            print(f"  {op}: {count}")
            
        print("\nObject Counts (Top 5):")
        for obj_count, freq in self.task_statistics['object_counts'].most_common(5):
            print(f"  {obj_count} objects: {freq}")
            
        print("\nTransformation Categories:")
        for category, count in self.task_statistics['transformation_categories'].most_common():
            print(f"  {category}: {count}")
            
        print(f"\nComplexity Score: Mean={np.mean(self.task_statistics['complexity_scores']):.2f}, Max={np.max(self.task_statistics['complexity_scores']):.2f}")
    
    def plot_statistics(self, filename=None):
        """Plot statistics about the analyzed tasks."""
        plt.figure(figsize=(15, 12))
        
        # 1. Grid Sizes
        plt.subplot(2, 3, 1)
        sizes = [eval(size) for size in self.task_statistics['grid_sizes'].keys()]
        heights = [size[0] for size in sizes]
        widths = [size[1] for size in sizes]
        counts = list(self.task_statistics['grid_sizes'].values())
        
        plt.scatter(widths, heights, s=[c*5 for c in counts], alpha=0.7)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Grid Sizes (bubble size = frequency)')
        plt.grid(True)
        
        # 2. Color Usage
        plt.subplot(2, 3, 2)
        colors = list(self.task_statistics['color_counts'].keys())
        color_counts = list(self.task_statistics['color_counts'].values())
        plt.bar(colors, color_counts, color=[ARC_CMAP_LIST[c] for c in colors])
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.title('Color Usage')
        
        # 3. Operation Frequency
        plt.subplot(2, 3, 3)
        ops = [op for op, _ in self.task_statistics['operation_counts'].most_common(10)]
        op_counts = [count for _, count in self.task_statistics['operation_counts'].most_common(10)]
        plt.barh(ops, op_counts)
        plt.xlabel('Count')
        plt.title('Top 10 Operations')
        
        # 4. Object Counts
        plt.subplot(2, 3, 4)
        obj_counts = [count for count, _ in self.task_statistics['object_counts'].most_common(10)]
        obj_freqs = [freq for _, freq in self.task_statistics['object_counts'].most_common(10)]
        plt.barh(obj_counts, obj_freqs)
        plt.xlabel('Frequency')
        plt.ylabel('Number of Objects')
        plt.title('Object Count Distribution (Top 10)')
        
        # 5. Transformation Categories
        plt.subplot(2, 3, 5)
        categories = list(self.task_statistics['transformation_categories'].keys())
        category_counts = list(self.task_statistics['transformation_categories'].values())
        plt.pie(category_counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Transformation Categories')
        
        # 6. Complexity Distribution
        plt.subplot(2, 3, 6)
        plt.hist(self.task_statistics['complexity_scores'], bins=20)
        plt.xlabel('Complexity Score')
        plt.ylabel('Number of Tasks')
        plt.title('Task Complexity Distribution')
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
        
    def find_similar_tasks(self, operations, top_n=5):
        """Find tasks that use similar operations."""
        similarities = {}
        for task_id, ops in self.task_operations.items():
            # Calculate Jaccard similarity
            op_set = set(ops)
            op_query = set(operations)
            
            if not op_set or not op_query:
                continue
                
            intersection = len(op_set.intersection(op_query))
            union = len(op_set.union(op_query))
            similarity = intersection / union if union > 0 else 0
            
            similarities[task_id] = similarity
            
        # Get top N similar tasks
        top_tasks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_tasks
        
    def recommend_tasks_by_difficulty(self, difficulty_level='medium', n=5):
        """Recommend tasks based on difficulty level."""
        # Map difficulty levels to complexity score ranges
        difficulty_ranges = {
            'easy': (0, 2),
            'medium': (2, 4),
            'hard': (4, float('inf'))
        }
        
        low, high = difficulty_ranges.get(difficulty_level, (0, float('inf')))
        
        # Find tasks with complexity in the specified range
        matching_tasks = []
        for task_id in self.task_operations.keys():
            try:
                complexity = np.mean([self.task_statistics['complexity_scores'][i] 
                                      for i, t in enumerate(self.task_operations.keys()) if t == task_id])
                if low <= complexity < high:
                    matching_tasks.append((task_id, complexity))
            except:
                continue
                
        # Sort by complexity
        matching_tasks.sort(key=lambda x: x[1])
        
        # Return top N tasks
        return matching_tasks[:n]

def main():
    # Example usage
    analyzer = ARCTaskAnalyzer()
    
    # Define paths to ARC datasets
    dataset_dirs = [
        "dataset/arc-dataset-diva/data",
        "dataset/arc-dataset-tama",
        "dataset/Mini-ARC"
    ]
    
    # Analyze dataset (limit to 500 tasks for quicker analysis)
    task_summaries = analyzer.analyze_dataset(dataset_dirs, max_tasks=500)
    
    # Print statistics
    analyzer.print_statistics()
    
    # Plot statistics
    analyzer.plot_statistics("arc_dataset_analysis.png")
    
    # Examples of using the analyzer
    print("\n=== Example Queries ===\n")
    
    # Find tasks similar to color_change and object_addition
    similar_tasks = analyzer.find_similar_tasks(['color_change', 'object_addition'])
    print("Tasks Similar to 'color_change' and 'object_addition':")
    for task_id, similarity in similar_tasks:
        print(f"  {task_id}: similarity = {similarity:.2f}")
    
    # Recommend medium difficulty tasks
    medium_tasks = analyzer.recommend_tasks_by_difficulty('medium')
    print("\nRecommended Medium Difficulty Tasks:")
    for task_id, complexity in medium_tasks:
        print(f"  {task_id}: complexity = {complexity:.2f}")

if __name__ == "__main__":
    main() 