# DQN Card Game

A sophisticated AI-powered card game platform that trains a model using Deep Q-Learning, evaluates its performance, and visualizes results through an interactive web dashboard built with React and Tailwind CSS.

![Card Game Banner](assets/banner.png)

## Project Overview

DQN Card Game is a comprehensive AI project that demonstrates deep reinforcement learning applied to a collectible card game. The project combines game design, AI model training, data analysis, and web visualization to create a complete end-to-end solution.

## Key Features

- **Custom Card Game Engine**: A fully-featured card game with deck building, various card abilities, and strategic gameplay
- **Deep Q-Network (DQN) Implementation**: AI agent trained using reinforcement learning to develop strategic gameplay
- **Performance Analysis**: Detailed analysis of deck performance, card win rates, and AI agent behavior
- **Automated Deck Building**: ML-powered system that can suggest optimal card combinations
- **Interactive Visualization**: Web-based dashboard for exploring gameplay metrics and card performance data

## Technical Architecture

### Game Engine (Python)
- Custom card game implementation with comprehensive rules logic
- Object-oriented design with classes for cards, decks, game states, and player actions
- Flexible action space that supports various strategic gameplay options

### Reinforcement Learning (TensorFlow)
- Deep Q-Network (DQN) agent implementation
- Experience replay and target network for stable learning
- Epsilon-greedy exploration strategy
- Comprehensive state representation capturing game complexity

### Deck Analysis System
- ML-based analysis of card synergies and deck performance
- Win rate tracking for individual cards and complete decks
- Recommendation engine for deck optimization

### Web Dashboard (React + Tailwind CSS)
- Interactive visualization of training progress and game statistics
- Card performance explorer with filtering and sorting
- Deck builder with AI-assisted recommendations

## Project Structure

```
dqn-card-game/
├── game.py              # Core game engine and DQN agent implementation
├── phase2.py            # Deck analysis and performance evaluation system
├── web-dashboard/       # React-based web interface for visualization
├── models/              # Saved neural network models
├── data/                # Game logs and performance metrics
└── assets/              # Images and other static resources
```

## Deep Q-Learning Approach

The AI agent is trained using a Deep Q-Network (DQN) approach with several key enhancements:

1. **State Representation**: Game state is encoded as a 68-dimensional vector capturing card attributes, board state, player resources, and game progression.

2. **Action Selection**: The agent uses an epsilon-greedy strategy, balancing exploration of new strategies with exploitation of known effective moves.

3. **Reward Function**: Custom reward function that considers:
   - Game outcome (win/loss)
   - Board control
   - Resource efficiency 
   - Strategic positioning

4. **Neural Network Architecture**: Multi-layer perceptron with:
   - Input layer matching state dimensions
   - Two hidden layers with 64 neurons each using ReLU activation
   - Output layer with linear activation predicting Q-values for each action

## Performance Metrics

After training across 200 episodes:

- **Win Rate Against Rule-Based AI**: 76%
- **Average Reward per Episode**: 142.8
- **Successful Strategic Patterns**: The agent learned to prioritize board control, efficiently manage resources, and identify powerful card combinations.

## Future Enhancements

- Implement Monte Carlo Tree Search (MCTS) for comparison with DQN
- Add multiplayer capabilities for human vs. AI gameplay
- Expand card database and abilities for more strategic depth
- Develop a mobile client for on-the-go play

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.4+
- Node.js 14+ (for web dashboard)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dqn-card-game.git
   cd dqn-card-game
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install web dashboard dependencies:
   ```
   cd web-dashboard
   npm install
   ```

### Running the Game Engine

```
python game.py
```

### Training the AI Agent

```
python game.py --train --episodes 200
```

### Running the Deck Analysis

```
python phase2.py
```

### Starting the Web Dashboard

```
cd web-dashboard
npm start
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Gym](https://gym.openai.com/) for inspiration on environment design
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [Hearthstone](https://playhearthstone.com/) and [Magic: The Gathering](https://magic.wizards.com/) for card game mechanics inspiration
