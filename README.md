# AlphaUP - Advanced Game AI with MCTS and Deep Learning

## Overview
AlphaUP is an implementation of Monte Carlo Tree Search (MCTS) combined with deep learning techniques, inspired by the principles used in AlphaGo. The project demonstrates the practical application of advanced AI techniques in decision-making systems.

## Key Features
- Monte Carlo Tree Search implementation
- Deep Residual Neural Network (ResNet) architecture
- CUDA-optimized tensor operations
- State-space exploration and evaluation
- Policy and value network heads
- Efficient tree traversal and expansion

## Technical Architecture

### 1. Neural Network Components
- **ResNet Architecture**
  - Custom residual blocks
  - Policy head for action probability distribution
  - Value head for position evaluation
  - Batch normalization layers
  - ReLU activation functions

### 2. MCTS Implementation
- **Tree Node Structure**
  - Visit count tracking
  - Action value estimation
  - Prior probability integration
  - UCB-based selection

- **Search Algorithm**
  - Efficient node expansion
  - Backpropagation of values
  - Dynamic exploration vs exploitation balance

### 3. Game Environment
- **State Representation**
  - Efficient board encoding
  - Valid move generation
  - State transition functions
  - Win condition evaluation

## Implementation Details

### Neural Network Architecture
```python
class ResNet(nn.Module):
    def __init__(self, game, num_residual_blocks, num_hidden):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backbone = nn.ModuleList(
            [ResidualBlock(num_hidden) for _ in range(num_residual_blocks)]
        )
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rows * game.cols, game.action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rows * game.cols, 1),
            nn.Tanh()
        )
```

### MCTS Implementation
```python
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)
        
        for _ in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select_child()
                
            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action
            )
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)
            
        return self.get_action_probabilities(root)
```

## Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- NumPy

## Installation
```bash
pip install torch numpy
```

## Usage Example
```python
# Initialize game environment
game = GameEnvironment()

# Setup MCTS parameters
args = {
    'C': 1.41,
    'num_searches': 1000
}

# Create MCTS instance
mcts = MCTS(game, args)

# Run search from current state
state = game.get_initial_state()
action_probabilities = mcts.search(state)
```

## Future Improvements
- Self-play training implementation
- Parallel MCTS search
- Temperature-based exploration
- Progressive widening
- Neural network architecture optimization

## Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
