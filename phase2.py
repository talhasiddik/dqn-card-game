import json
import logging
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from game import Card, CardDatabase,CardGameEnv, Deck, GameState, DQNAgent

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("deck_analysis.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Constants
MAX_BOARD_SIZE = 5
MAX_HAND_SIZE = 7
MAX_LORE = 20
DECK_SIZE = 30

class DeckAnalyzer:
    def __init__(self, card_database: CardDatabase):
        self.card_db = card_database
        self.match_history = []
        self.deck_performance = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.card_performance = defaultdict(lambda: {"wins": 0, "losses": 0, "played": 0})
    
    def load_match_history(self, filename="match_history.json"):
        """Load match history from a file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.match_history = json.load(f)
                logger.info(f"Loaded {len(self.match_history)} matches from history")
            except Exception as e:
                logger.error(f"Error loading match history: {e}")
        else:
            logger.warning(f"Match history file {filename} not found.")
    
    def save_match_history(self, filename="match_history.json"):
        """Save match history to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.match_history, f, indent=2)
            logger.info(f"Saved {len(self.match_history)} matches to history")
        except Exception as e:
            logger.error(f"Error saving match history: {e}")
    
    def add_match_result(self, deck: List[int], winner: int, match_data: Dict):
        """Add a new match result to the history."""
        match_entry = {
            "deck": deck,
            "winner": winner,
            "data": match_data
        }
        self.match_history.append(match_entry)
        
        # Update deck performance
        deck_key = tuple(sorted(deck))
        if winner == 0:
            self.deck_performance[deck_key]["wins"] += 1
        else:
            self.deck_performance[deck_key]["losses"] += 1
        
        # Update card performance
        for card_id in deck:
            if winner == 0:
                self.card_performance[card_id]["wins"] += 1
            else:
                self.card_performance[card_id]["losses"] += 1
            self.card_performance[card_id]["played"] += 1
    
    def get_deck_win_rate(self, deck: List[int]) -> float:
        """Calculate the win rate of a specific deck."""
        deck_key = tuple(sorted(deck))
        if deck_key not in self.deck_performance:
            return 0.0
        wins = self.deck_performance[deck_key]["wins"]
        losses = self.deck_performance[deck_key]["losses"]
        return wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    def get_card_win_rate(self, card_id: int) -> float:
        """Calculate the win rate of a specific card."""
        if card_id not in self.card_performance:
            return 0.0
        wins = self.card_performance[card_id]["wins"]
        losses = self.card_performance[card_id]["losses"]
        return wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    def recommend_deck_improvements(self, deck: List[int]) -> List[Tuple[int, float]]:
        """Recommend which cards to replace in a deck based on win rates."""
        card_win_rates = []
        for card_id in deck:
            win_rate = self.get_card_win_rate(card_id)
            card_win_rates.append((card_id, win_rate))
        
        # Sort cards by win rate (ascending)
        card_win_rates.sort(key=lambda x: x[1])
        
        # Recommend replacing the worst-performing cards
        return card_win_rates[:3]  # Top 3 cards to replace
    
    
    
    def predict_best_deck(self, num_decks: int = 5) -> List[Tuple[List[int], float]]:
        """Predict the best deck compositions based on match history."""
        # Extract deck features and win rates
        deck_features = []
        win_rates = []
    
        for deck_key, performance in self.deck_performance.items():
            deck_features.append(list(deck_key))
            win_rate = performance["wins"] / (performance["wins"] + performance["losses"])
            win_rates.append(win_rate)
    
        # Check if we have enough data to split
        if len(deck_features) <= 1:
            logger.warning("Not enough data to predict best decks.")
            return []
    
        # Train a simple model to predict win rates
        X = np.array(deck_features)
        y = np.array(win_rates)
    
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        # Predict win rates for all possible decks
        all_decks = self.generate_all_decks()
        predicted_win_rates = model.predict(all_decks)
    
        # Sort decks by predicted win rate
        sorted_decks = sorted(zip(all_decks, predicted_win_rates), key=lambda x: x[1], reverse=True)
    
        # Return the top N decks
        return sorted_decks[:num_decks]
    
    def generate_all_decks(self) -> List[List[int]]:
        """Generate all possible deck combinations (for demonstration purposes)."""
        # This is a simplified version; in practice, you would use a more efficient method
        all_cards = list(self.card_db.cards.keys())
        return [random.sample(all_cards, DECK_SIZE) for _ in range(1000)]  # Generate 1000 random decks
    
    def visualize_deck_performance(self):
        """Visualize deck performance using matplotlib."""
        deck_win_rates = []
        for deck_key, performance in self.deck_performance.items():
            win_rate = performance["wins"] / (performance["wins"] + performance["losses"])
            deck_win_rates.append(win_rate)
        
        plt.figure(figsize=(10, 5))
        plt.hist(deck_win_rates, bins=20, edgecolor='black')
        plt.title("Deck Win Rate Distribution")
        plt.xlabel("Win Rate")
        plt.ylabel("Frequency")
        plt.savefig("deck_win_rate_distribution.png")
        plt.close()


def load_trained_model(state_size, action_size, model_weights_path="dqn_agent_final.weights.h5"):
    """Load the trained DQN model from saved weights."""
    agent = DQNAgent(state_size, action_size)
    agent.model = keras.Sequential([
        keras.layers.Dense(64, input_dim=state_size, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')
    ])
    agent.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=agent.learning_rate))
    agent.load(model_weights_path)
    agent.epsilon = 0.0  # Disable exploration (use the trained model for predictions)
    return agent

# Add this function to your phase2.py file, just before the simulate_matches function
def convert_state_to_input(state_dict):
    """
    Convert the game state dictionary into a numerical format with exactly 68 features.
    Carefully tailored to match the original model's input shape.
    
    Args:
        state_dict: Dictionary representation of the game state
        
    Returns:
        numpy array: Flattened numerical representation of the state with exactly 68 features
    """
    # Initialize an empty array to hold the state features
    state_array = []
    
    # Basic game state (2 features)
    state_array.append(state_dict['turn_number'])
    state_array.append(state_dict['current_player'])
    
    # Player state (5 features)
    player = state_dict['player']
    state_array.append(player['player_id'])
    state_array.append(player['lore'])
    state_array.append(player['health'])
    state_array.append(player['deck_size'])
    state_array.append(len(player['hand']))
    
    # Opponent state (5 features)
    opponent = state_dict['opponent']
    state_array.append(opponent['player_id'])
    state_array.append(opponent['lore'])
    state_array.append(opponent['health'])
    state_array.append(opponent['deck_size'])
    state_array.append(len(opponent['hand']))
    
    # Board state (for both players - 28 features per player = 56 total)
    # For player board (28 features)
    for i in range(4):  # Use 4 board slots instead of 5
        if i < len(state_dict['player_board']) and state_dict['player_board'][i] is not None:
            card = state_dict['player_board'][i]
            # Each card has 7 features
            state_array.extend([1, card['id'] % 50, card['cost'], card['attack'], card['defense'], 
                              1 if 'rush' in card.get('abilities', []) else 0,
                              1 if 'taunt' in card.get('abilities', []) else 0])
        else:
            state_array.extend([0, 0, 0, 0, 0, 0, 0])
    
    # For opponent board (28 features)
    for i in range(4):  # Use 4 board slots instead of 5
        if i < len(state_dict['opponent_board']) and state_dict['opponent_board'][i] is not None:
            card = state_dict['opponent_board'][i]
            # Each card has 7 features
            state_array.extend([1, card['id'] % 50, card['cost'], card['attack'], card['defense'], 
                              1 if 'rush' in card.get('abilities', []) else 0,
                              1 if 'taunt' in card.get('abilities', []) else 0])
        else:
            state_array.extend([0, 0, 0, 0, 0, 0, 0])
    
    # Ensure we have exactly 68 features
    if len(state_array) != 68:
        # Add padding or truncate if necessary
        if len(state_array) < 68:
            state_array.extend([0] * (68 - len(state_array)))
        else:
            state_array = state_array[:68]
    
    # Final check
    assert len(state_array) == 68, f"Expected 68 features, but got {len(state_array)}"
    
    # Convert to numpy array and reshape for the model
    return np.array([state_array], dtype=np.float32)  # Ensure float type for the model

# Now modify the simulate_matches function to use this conversion
def simulate_matches(analyzer: DeckAnalyzer, card_db: CardDatabase, agent, num_matches: int = 100):
    """Simulate matches using the trained DQN agent to populate the match history."""
    env = CardGameEnv(card_db)
    logger.info(f"Simulating {num_matches} matches using the trained DQN agent...")
    
    for match_num in range(num_matches):
        logger.info(f"Starting match {match_num+1}/{num_matches}")
        
        # Generate a random deck for the player
        player_deck = [card.id for card in card_db.get_random_cards(DECK_SIZE)]
        player_deck_obj = Deck([Card(id=card_id, name=f"Card {card_id}", cost=1, attack=1, defense=1, abilities=[]) for card_id in player_deck])
        
        # Generate a random deck for the opponent
        opponent_deck = [card.id for card in card_db.get_random_cards(DECK_SIZE)]
        opponent_deck_obj = Deck([Card(id=card_id, name=f"Card {card_id}", cost=1, attack=1, defense=1, abilities=[]) for card_id in opponent_deck])
        
        # Initialize the game state
        game_state = GameState(player_deck_obj, opponent_deck_obj)
        game_state.initialize_game()
        
        # Initialize game environment
        env.reset()
        env.game_state = game_state
        
        # Add a turn counter and maximum turns to prevent infinite loops
        turn_counter = 0
        max_turns = 100  # Adjust as needed
        
        # Simulate the match using the trained DQN agent
        while not game_state.game_over and turn_counter < max_turns:
            # Increment turn counter
            turn_counter += 1
            
            if turn_counter % 10 == 0:
                logger.info(f"  Match {match_num+1}: Turn {turn_counter}")
            
            # Get the current state as a dictionary
            state_dict = game_state.get_state()
            
            # Convert the state dictionary to a format the model can use
            state_array = convert_state_to_input(state_dict)
            
            # Get valid actions for the current player
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                logger.warning(f"  Match {match_num+1}: No valid actions available on turn {turn_counter}")
                break
                
            # Use the trained DQN agent to select an action
            try:
                action = agent.act(state_array, valid_actions)
                logger.debug(f"  Match {match_num+1}: Selected action {action}")
            except Exception as e:
                logger.error(f"  Match {match_num+1}: Error selecting action: {e}")
                break
            
            # Execute the action
            try:
                step_result = env.step(action)
                if len(step_result) == 3:
                    reward, next_state, done = step_result
                elif len(step_result) == 4:
                    reward, next_state, done, _ = step_result
                else:
                    reward = step_result[0] if len(step_result) > 0 else 0
                    done = step_result[2] if len(step_result) > 2 else False
                
                logger.debug(f"  Match {match_num+1}: Action result - reward: {reward}, done: {done}")
            except Exception as e:
                logger.error(f"  Match {match_num+1}: Error executing step: {e}")
                break
            
            # Check if the game is over
            if done:
                logger.info(f"  Match {match_num+1}: Game over on turn {turn_counter}")
                break
        
        # Check if we reached max turns
        if turn_counter >= max_turns:
            logger.warning(f"  Match {match_num+1}: Reached maximum turns ({max_turns}), forcing game end")
            game_state.game_over = True
            game_state.winner = random.randint(0, 1)  # Randomly assign winner
        
        # Add the match result to the analyzer
        winner = game_state.winner
        analyzer.add_match_result(player_deck, winner, match_data={"turns": turn_counter})
        logger.info(f"  Match {match_num+1}: Completed - Winner: Player {winner}")
    
    logger.info(f"Added {num_matches} matches to history")
    
# Main function for Phase 2
def main():
    # Load card database
    card_db = CardDatabase("cards.json")
    
    # Initialize deck analyzer
    analyzer = DeckAnalyzer(card_db)
    analyzer.load_match_history("match_history.json")
    
    # Load the trained DQN agent
    state_size = 68  # Adjust based on your state representation size
    action_size = 7   # Number of action types
    agent = load_trained_model(state_size, action_size, "dqn_agent_final.weights.h5")
    
    # Simulate matches using the trained DQN agent
    simulate_matches(analyzer, card_db, agent, num_matches=10)
    
    # Save updated match history
    analyzer.save_match_history("match_history.json")
    
    # Example: Get deck win rate
    example_deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # actual card IDs
    win_rate = analyzer.get_deck_win_rate(example_deck)
    logger.info(f"Deck win rate: {win_rate:.2f}")
    
    # Example: Recommend deck improvements
    recommendations = analyzer.recommend_deck_improvements(example_deck)
    logger.info("Recommended cards to replace:")
    for card_id, win_rate in recommendations:
        card = card_db.get_card(card_id)
        logger.info(f"{card.name} (Win Rate: {win_rate:.2f})")
    
    # Example: Predict best decks
    best_decks = analyzer.predict_best_deck(num_decks=5)
    logger.info("Top 5 predicted decks:")
    for deck, win_rate in best_decks:
        logger.info(f"Deck: {deck}, Predicted Win Rate: {win_rate:.2f}")
    with open("predicted_decks.json", "w") as f:
        json.dump([{"deck": deck, "predicted_win_rate": win_rate} for deck, win_rate in best_decks], f, indent=4)
        
    # Visualize deck performance
    analyzer.visualize_deck_performance()
        
if __name__ == "__main__":
    main()