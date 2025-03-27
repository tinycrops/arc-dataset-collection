# ARC Task Solver - Jellyfish-Inspired Model

A neural-symbolic system for solving tasks from the Abstraction and Reasoning Corpus (ARC). This system is inspired by jellyfish nervous systems and combines a neural controller with symbolic operations to solve ARC tasks.

## System Architecture

The model consists of four main components:

1. **Sensory System (Rhopalia)**: Multiple sensory modules that extract different features from the input grids:
   - Grid size and shape
   - Color distribution
   - Local structure patterns
   - Object detection

2. **Neural Controller**: A multi-layer perceptron that processes the sensory input and selects the appropriate transformation action.

3. **Symbolic Operations**: A set of predefined grid transformations like rotations, flips, and color changes.

4. **Training System**: A pipeline that analyzes the dataset, infers which transformations are required, and trains the controller to select the right operation.

## Files

- `arc_model.py`: Core model architecture including the sensory system, controller, and symbolic operations
- `arc_trainer.py`: Dataset loading, training, and evaluation system
- `arc_controller.pth`: Trained model weights

## Usage

1. **Training a new model**:
   ```
   python arc_trainer.py
   ```
   This will analyze the ARC dataset, identify common operations, train a model, and save the weights.

2. **Using a trained model**:
   ```python
   from arc_model import JellyfishARCSystem
   import torch
   import numpy as np
   
   # Load model
   model = JellyfishARCSystem()
   model.controller.load_state_dict(torch.load('arc_controller.pth'))
   
   # Process a grid
   input_grid = np.array([[0, 1], [2, 3]])
   output_grid, action_name = model.process_grid(input_grid)
   print(f"Selected action: {action_name}")
   print(output_grid)
   ```

## Performance

The model can identify and learn simple transformations like rotations, flips, and shifts. For more complex operations, the system would need to be extended with additional symbolic operations.

## Limitations and Future Work

- The current system is limited to the predefined set of symbolic operations
- More complex ARC tasks require composition of operations
- Future improvements could include:
  - Learning to compose multiple operations
  - Adding more advanced grid transformations
  - Self-supervised discovery of new operations

## References

- Abstraction and Reasoning Corpus (ARC): https://github.com/fchollet/ARC
- Inspired by research on jellyfish nervous systems and hybrid neural-symbolic approaches to AI 