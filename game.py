import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union
import time
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("game_simulation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Constants
MAX_HAND_SIZE = 7
MAX_BOARD_SIZE = 5
MAX_LORE = 20
INKWELL_START_SIZE = 3

# Card definitions
@dataclass
class Card:
    id: int
    name: str
    cost: int
    attack: int
    defense: int
    abilities: List[str]
    lore_value: int = 0
    song_effect: str = ""
    
    def __str__(self):
        return f"{self.name} ({self.cost}) {self.attack}/{self.defense}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'cost': self.cost,
            'attack': self.attack,
            'defense': self.defense,
            'abilities': self.abilities,
            'lore_value': self.lore_value,
            'song_effect': self.song_effect
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            name=data['name'],
            cost=data['cost'],
            attack=data['attack'],
            defense=data['defense'],
            abilities=data['abilities'],
            lore_value=data.get('lore_value', 0),
            song_effect=data.get('song_effect', "")
        )


class CardDatabase:
    def __init__(self, cards_file=None):
        self.cards = {}
        if cards_file and os.path.exists(cards_file):
            self.load_cards(cards_file)
        else:
            self.generate_sample_cards()
    
    def load_cards(self, filename):
        try:
            with open(filename, 'r') as f:
                cards_data = json.load(f)
                for card_data in cards_data:
                    card = Card.from_dict(card_data)
                    self.cards[card.id] = card
            logger.info(f"Loaded {len(self.cards)} cards from database")
        except Exception as e:
            logger.error(f"Error loading cards: {e}")
            self.generate_sample_cards()
    
    def save_cards(self, filename):
        try:
            cards_data = [card.to_dict() for card in self.cards.values()]
            with open(filename, 'w') as f:
                json.dump(cards_data, f, indent=2)
            logger.info(f"Saved {len(self.cards)} cards to database")
        except Exception as e:
            logger.error(f"Error saving cards: {e}")
    
    def generate_sample_cards(self):
        # Generate sample cards for testing
        abilities = ["Swift", "Guard", "Flying", "Stealth", "Poison", "Shield", "Draw"]
        song_effects = ["Draw a card", "Gain 2 lore", "Deal 1 damage to all enemies", 
                       "Heal all allies 1", "Reduce cost of next card by 2"]
        
        for i in range(1, 51):
            cost = random.randint(1, 5)
            attack = random.randint(0, cost * 2)
            defense = random.randint(1, cost * 2)
            card_abilities = random.sample(abilities, k=min(random.randint(0, 2), len(abilities)))
            
            lore_value = random.randint(0, 2)
            song_effect = random.choice(song_effects) if random.random() > 0.7 else ""
            
            card = Card(
                id=i,
                name=f"Card {i}",
                cost=cost,
                attack=attack,
                defense=defense,
                abilities=card_abilities,
                lore_value=lore_value,
                song_effect=song_effect
            )
            self.cards[i] = card
        
        logger.info(f"Generated {len(self.cards)} sample cards")
    
    def get_card(self, card_id):
        return self.cards.get(card_id)
    
    def get_random_cards(self, num_cards):
        card_ids = list(self.cards.keys())
        selected_ids = random.sample(card_ids, min(num_cards, len(card_ids)))
        return [self.cards[card_id] for card_id in selected_ids]
    
    def get_all_cards(self):
        return list(self.cards.values())


class Deck:
    def __init__(self, cards: List[Card]):
        self.initial_cards = cards.copy()
        self.cards = cards.copy()
        self.discard_pile = []
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        if not self.cards and self.discard_pile:
            self.cards = self.discard_pile
            self.discard_pile = []
            self.shuffle()
            logger.debug("Reshuffled discard pile into deck")
        
        if not self.cards:
            return None
        
        return self.cards.pop()
    
    def discard(self, card: Card):
        self.discard_pile.append(card)
    
    def reset(self):
        self.cards = self.initial_cards.copy()
        self.discard_pile = []
        self.shuffle()
    
    def remaining(self) -> int:
        return len(self.cards)
    
    def __len__(self):
        return len(self.cards)


class GameBoard:
    def __init__(self):
        self.player_board = [None] * MAX_BOARD_SIZE
        self.opponent_board = [None] * MAX_BOARD_SIZE
        self.player_tapped = [False] * MAX_BOARD_SIZE
        self.opponent_tapped = [False] * MAX_BOARD_SIZE
    
    def play_card(self, player_id: int, card: Card, position: int) -> bool:
        """Play card to the specified position. Returns True if successful."""
        if player_id == 0:  # Player
            if position < 0 or position >= MAX_BOARD_SIZE or self.player_board[position] is not None:
                return False
            self.player_board[position] = card
            self.player_tapped[position] = False
        else:  # Opponent
            if position < 0 or position >= MAX_BOARD_SIZE or self.opponent_board[position] is not None:
                return False
            self.opponent_board[position] = card
            self.opponent_tapped[position] = False
        return True
    
    def tap_card(self, player_id: int, position: int) -> bool:
        """Tap a card at the specified position. Returns True if successful."""
        if player_id == 0:  # Player
            if (position < 0 or position >= MAX_BOARD_SIZE or 
                    self.player_board[position] is None or self.player_tapped[position]):
                return False
            self.player_tapped[position] = True
        else:  # Opponent
            if (position < 0 or position >= MAX_BOARD_SIZE or 
                    self.opponent_board[position] is None or self.opponent_tapped[position]):
                return False
            self.opponent_tapped[position] = True
        return True
    
    def attack(self, player_id: int, attacker_pos: int, target_pos: int) -> Tuple[bool, int, Card, Card]:
        """
        Execute an attack. Returns (success, damage, attacker, target).
        Damage > 0 means the target was destroyed.
        """
        if player_id == 0:  # Player attacking
            if (attacker_pos < 0 or attacker_pos >= MAX_BOARD_SIZE or 
                    self.player_board[attacker_pos] is None or self.player_tapped[attacker_pos]):
                return False, 0, None, None
            
            if target_pos < 0 or target_pos >= MAX_BOARD_SIZE or self.opponent_board[target_pos] is None:
                return False, 0, None, None
            
            attacker = self.player_board[attacker_pos]
            target = self.opponent_board[target_pos]
            
            # Apply attack damage
            damage = max(0, attacker.attack - target.defense)
            
            if damage >= target.defense:
                # Target is destroyed
                self.opponent_board[target_pos] = None
            
            # Attacker is tapped after attacking
            self.player_tapped[attacker_pos] = True
            
            return True, damage, attacker, target
        
        else:  # Opponent attacking
            if (attacker_pos < 0 or attacker_pos >= MAX_BOARD_SIZE or 
                    self.opponent_board[attacker_pos] is None or self.opponent_tapped[attacker_pos]):
                return False, 0, None, None
            
            if target_pos < 0 or target_pos >= MAX_BOARD_SIZE or self.player_board[target_pos] is None:
                return False, 0, None, None
            
            attacker = self.opponent_board[attacker_pos]
            target = self.player_board[target_pos]
            
            # Apply attack damage
            damage = max(0, attacker.attack - target.defense)
            
            if damage >= target.defense:
                # Target is destroyed
                self.player_board[target_pos] = None
            
            # Attacker is tapped after attacking
            self.opponent_tapped[attacker_pos] = True
            
            return True, damage, attacker, target
    
    def sing(self, player_id: int, card_pos: int) -> Tuple[bool, str]:
        """Activate a card's song effect. Returns (success, effect)."""
        if player_id == 0:  # Player
            if (card_pos < 0 or card_pos >= MAX_BOARD_SIZE or 
                    self.player_board[card_pos] is None or self.player_tapped[card_pos]):
                return False, ""
            
            card = self.player_board[card_pos]
            if not card.song_effect:
                return False, ""
            
            effect = card.song_effect
            self.player_tapped[card_pos] = True
            return True, effect
        
        else:  # Opponent
            if (card_pos < 0 or card_pos >= MAX_BOARD_SIZE or 
                    self.opponent_board[card_pos] is None or self.opponent_tapped[card_pos]):
                return False, ""
            
            card = self.opponent_board[card_pos]
            if not card.song_effect:
                return False, ""
            
            effect = card.song_effect
            self.opponent_tapped[card_pos] = True
            return True, effect
    
    def reset_tapped(self):
        """Reset all tapped states at the start of a new turn."""
        self.player_tapped = [False] * MAX_BOARD_SIZE
        self.opponent_tapped = [False] * MAX_BOARD_SIZE
    
    def get_state(self) -> Tuple[List[Optional[Card]], List[bool], List[Optional[Card]], List[bool]]:
        """Return the current board state."""
        return (
            self.player_board.copy(),
            self.player_tapped.copy(),
            self.opponent_board.copy(),
            self.opponent_tapped.copy()
        )
    
    def count_cards(self, player_id: int) -> int:
        """Count the number of cards a player has on the board."""
        if player_id == 0:
            return sum(1 for card in self.player_board if card is not None)
        else:
            return sum(1 for card in self.opponent_board if card is not None)


class Player:
    def __init__(self, player_id: int, deck: Deck):
        self.player_id = player_id
        self.deck = deck
        self.hand = []
        self.inkwell = []
        self.lore = 0
        self.health = 20
    
    def draw_card(self) -> Optional[Card]:
        """Draw a card from the deck to hand. Returns the drawn card or None if deck is empty."""
        if len(self.hand) >= MAX_HAND_SIZE:
            return None
        
        card = self.deck.draw()
        if card:
            self.hand.append(card)
        return card
    
    def fill_inkwell(self):
        """Add an ink to the inkwell at the start of turn (up to max)."""
        if len(self.inkwell) < INKWELL_START_SIZE + self.lore // 5:
            self.inkwell.append(True)
    
    def reset_inkwell(self):
        """Reset the inkwell for a new turn."""
        self.inkwell = [True] * len(self.inkwell)
    
    def can_play_card(self, card_index: int) -> bool:
        """Check if the player can play the card at the given index."""
        if card_index < 0 or card_index >= len(self.hand):
            return False
        
        card = self.hand[card_index]
        available_ink = sum(1 for ink in self.inkwell if ink)
        return card.cost <= available_ink
    
    def play_card(self, card_index: int) -> Optional[Card]:
        """
        Remove a card from hand and use inkwell. Returns the card if successful.
        The game will need to add the card to the board.
        """
        if not self.can_play_card(card_index):
            return None
        
        card = self.hand[card_index]
        
        # Use inkwell
        cost = card.cost
        for i in range(len(self.inkwell)):
            if cost > 0 and self.inkwell[i]:
                self.inkwell[i] = False
                cost -= 1
        
        # Remove card from hand
        self.hand.pop(card_index)
        
        return card
    
    def gain_lore(self, amount: int = 1):
        """Gain lore points."""
        self.lore = min(MAX_LORE, self.lore + amount)
    
    def take_damage(self, amount: int) -> bool:
        """
        Apply damage to player health. Returns True if player is defeated.
        """
        self.health -= amount
        return self.health <= 0
    
    def get_state(self) -> Dict:
        """Return the current player state."""
        return {
            'player_id': self.player_id,
            'hand': [card.to_dict() if card else None for card in self.hand],
            'inkwell': self.inkwell.copy(),
            'lore': self.lore,
            'health': self.health,
            'deck_size': len(self.deck.cards),
            'discard_size': len(self.deck.discard_pile)
        }


class GameState:
    """Main game state tracking class"""
    
    def __init__(self, player_deck: Deck, opponent_deck: Deck):
        self.board = GameBoard()
        self.player = Player(0, player_deck)
        self.opponent = Player(1, opponent_deck)
        self.current_player_id = 0  # 0 for player, 1 for opponent
        self.turn_number = 1
        self.game_over = False
        self.winner = None
        self.history = []
    
    def initialize_game(self):
        """Set up the initial game state."""
        # Shuffle decks
        self.player.deck.shuffle()
        self.opponent.deck.shuffle()
        
        # Draw initial hands (4 cards each)
        for _ in range(4):
            self.player.draw_card()
            self.opponent.draw_card()
        
        # Set initial inkwells
        self.player.inkwell = [True] * INKWELL_START_SIZE
        self.opponent.inkwell = [True] * INKWELL_START_SIZE
        
        # Randomize first player
        self.current_player_id = random.randint(0, 1)
        
        # Initial game state
        self.turn_number = 1
        self.game_over = False
        self.winner = None
    
    def start_turn(self):
        """Start a new turn for the current player."""
        current_player = self.player if self.current_player_id == 0 else self.opponent
        
        # Draw a card
        current_player.draw_card()
        
        # Reset tapped status for all cards
        self.board.reset_tapped()
        
        # Fill and reset inkwell
        current_player.fill_inkwell()
        current_player.reset_inkwell()
        
        self.log_state(f"Turn {self.turn_number} - Player {self.current_player_id} starts")
    
    def end_turn(self):
        """End the current turn and switch players."""
        # Switch players
        self.current_player_id = 1 - self.current_player_id
        
        # Increment turn number if the first player is about to play again
        if self.current_player_id == 0:
            self.turn_number += 1
        
        self.log_state(f"Turn {self.turn_number} - Player {self.current_player_id} is next")
    
    def play_card(self, card_index: int, board_position: int) -> bool:
        """Play a card from hand to the board at the specified position."""
        current_player = self.player if self.current_player_id == 0 else self.opponent
        
        # Check if the move is valid
        if not current_player.can_play_card(card_index):
            return False
        
        # Get the card from hand
        card = current_player.play_card(card_index)
        if not card:
            return False
        
        # Add to board
        success = self.board.play_card(self.current_player_id, card, board_position)
        if not success:
            # Put the card back in hand if it couldn't be played
            current_player.hand.append(card)
            return False
        
        # Gain lore if the card has lore value
        if card.lore_value > 0:
            current_player.gain_lore(card.lore_value)
        
        self.log_state(f"Player {self.current_player_id} played {card.name} at position {board_position}")
        return True
    
    def tap_card(self, board_position: int) -> bool:
        """Tap a card on the board."""
        success = self.board.tap_card(self.current_player_id, board_position)
        if success:
            self.log_state(f"Player {self.current_player_id} tapped card at position {board_position}")
        return success
    
    def attack(self, attacker_pos: int, target_pos: int) -> Tuple[bool, int]:
        """Execute an attack. Returns (success, damage)."""
        success, damage, attacker, target = self.board.attack(
            self.current_player_id, attacker_pos, target_pos)
        
        if success:
            self.log_state(
                f"Player {self.current_player_id} attacked with {attacker.name} "
                f"against {target.name}, dealing {damage} damage"
            )
            
            # If a card was destroyed, add lore
            if damage > 0:
                current_player = self.player if self.current_player_id == 0 else self.opponent
                current_player.gain_lore(1)
                self.log_state(f"Player {self.current_player_id} gained 1 lore from destroying a card")
        
        return success, damage
    
    def direct_attack(self, attacker_pos: int) -> bool:
        """Attack the opponent directly with a card."""
        current_player = self.player if self.current_player_id == 0 else self.opponent
        target_player = self.opponent if self.current_player_id == 0 else self.player
        
        if self.current_player_id == 0:
            # Player attacking
            if (attacker_pos < 0 or attacker_pos >= MAX_BOARD_SIZE or 
                    self.board.player_board[attacker_pos] is None or 
                    self.board.player_tapped[attacker_pos]):
                return False
            
            attacker = self.board.player_board[attacker_pos]
            
            # Check if there are any untapped cards on opponent board with "Guard" ability
            has_guard = any(
                card is not None and 
                not tapped and 
                "Guard" in card.abilities
                for card, tapped in zip(self.board.opponent_board, self.board.opponent_tapped)
            )
            
            if has_guard:
                return False
            
            # Apply damage to opponent
            damage = attacker.attack
            is_defeated = target_player.take_damage(damage)
            
            # Attacker is tapped after attacking
            self.board.player_tapped[attacker_pos] = True
            
            self.log_state(
                f"Player {self.current_player_id} attacked directly with {attacker.name}, "
                f"dealing {damage} damage to opponent"
            )
            
            if is_defeated:
                self.game_over = True
                self.winner = self.current_player_id
                self.log_state(f"Player {self.current_player_id} wins the game")
            
            return True
        
        else:
            # Opponent attacking
            if (attacker_pos < 0 or attacker_pos >= MAX_BOARD_SIZE or 
                    self.board.opponent_board[attacker_pos] is None or 
                    self.board.opponent_tapped[attacker_pos]):
                return False
            
            attacker = self.board.opponent_board[attacker_pos]
            
            # Check if there are any untapped cards on player board with "Guard" ability
            has_guard = any(
                card is not None and 
                not tapped and 
                "Guard" in card.abilities
                for card, tapped in zip(self.board.player_board, self.board.player_tapped)
            )
            
            if has_guard:
                return False
            
            # Apply damage to player
            damage = attacker.attack
            is_defeated = target_player.take_damage(damage)
            
            # Attacker is tapped after attacking
            self.board.opponent_tapped[attacker_pos] = True
            
            self.log_state(
                f"Player {self.current_player_id} attacked directly with {attacker.name}, "
                f"dealing {damage} damage to opponent"
            )
            
            if is_defeated:
                self.game_over = True
                self.winner = self.current_player_id
                self.log_state(f"Player {self.current_player_id} wins the game")
            
            return True
    
    def sing(self, card_pos: int) -> Tuple[bool, str]:
        """Activate a card's song effect."""
        success, effect = self.board.sing(self.current_player_id, card_pos)
        
        if success:
            current_player = self.player if self.current_player_id == 0 else self.opponent
            
            self.log_state(f"Player {self.current_player_id} activated song: {effect}")
            
            # Simple effect handling
            if "Draw a card" in effect:
                current_player.draw_card()
                self.log_state(f"Player {self.current_player_id} drew a card")
            
            elif "Gain 2 lore" in effect:
                current_player.gain_lore(2)
                self.log_state(f"Player {self.current_player_id} gained 2 lore")
            
            # More complex effects would be handled here
            
            # Give a small bonus for using songs
            current_player.gain_lore(1)
        
        return success, effect
    
    def gain_lore(self) -> bool:
        """Gain lore as an action."""
        current_player = self.player if self.current_player_id == 0 else self.opponent
        current_player.gain_lore(1)
        self.log_state(f"Player {self.current_player_id} gained 1 lore as an action")
        return True
    
    def check_win_condition(self) -> bool:
        """Check if the game is over."""
        if self.player.health <= 0:
            self.game_over = True
            self.winner = 1
            self.log_state("Player 1 (opponent) wins the game")
            return True
        
        if self.opponent.health <= 0:
            self.game_over = True
            self.winner = 0
            self.log_state("Player 0 (player) wins the game")
            return True
        
        # Check if a player cannot draw or play cards
        if (len(self.player.deck.cards) == 0 and 
                len(self.player.deck.discard_pile) == 0 and 
                len(self.player.hand) == 0 and 
                self.board.count_cards(0) == 0):
            self.game_over = True
            self.winner = 1
            self.log_state("Player 1 (opponent) wins - Player 0 has no cards left")
            return True
        
        if (len(self.opponent.deck.cards) == 0 and 
                len(self.opponent.deck.discard_pile) == 0 and 
                len(self.opponent.hand) == 0 and 
                self.board.count_cards(1) == 0):
            self.game_over = True
            self.winner = 0
            self.log_state("Player 0 (player) wins - Player 1 has no cards left")
            return True
        
        return False
    
    def get_state(self) -> Dict:
        """Return the complete game state."""
        player_board, player_tapped, opponent_board, opponent_tapped = self.board.get_state()
        
        return {
            'turn_number': self.turn_number,
            'current_player': self.current_player_id,
            'game_over': self.game_over,
            'winner': self.winner,
            'player': self.player.get_state(),
            'opponent': self.opponent.get_state(),
            'player_board': [card.to_dict() if card else None for card in player_board],
            'player_tapped': player_tapped,
            'opponent_board': [card.to_dict() if card else None for card in opponent_board],
            'opponent_tapped': opponent_tapped
        }
    
    def log_state(self, action_description: str = None):
        """Log the current game state and action."""
        state = self.get_state()
        if action_description:
            state['action'] = action_description
        state['timestamp'] = time.time()
        self.history.append(state)
        logger.debug(f"Game state: {action_description}")
    
    def save_game_history(self, filename: str):
        """Save the game history to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Game history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving game history: {e}")


# Define the action space
class ActionSpace:
    PLAY_CARD = 0
    TAP_CARD = 1
    ATTACK = 2
    DIRECT_ATTACK = 3
    SING = 4
    GAIN_LORE = 5
    END_TURN = 6
    
    @staticmethod
    def get_action_name(action_type: int) -> str:
        """Get the string representation of an action type."""
        action_names = {
            ActionSpace.PLAY_CARD: "Play Card",
            ActionSpace.TAP_CARD: "Tap Card",
            ActionSpace.ATTACK: "Attack",
            ActionSpace.DIRECT_ATTACK: "Direct Attack",
            ActionSpace.SING: "Sing",
            ActionSpace.GAIN_LORE: "Gain Lore",
            ActionSpace.END_TURN: "End Turn"
        }
        return action_names.get(action_type, "Unknown Action")


class GameAction:
    def __init__(self, action_type: int, params: Dict = None):
        self.action_type = action_type
        self.params = params or {}
    
    def execute(self, game_state: GameState) -> bool:
        """Execute the action on the game state. Returns success."""
        if game_state.game_over:
            return False
        
        if self.action_type == ActionSpace.PLAY_CARD:
            card_index = self.params.get('card_index', 0)
            board_position = self.params.get('board_position', 0)
            return game_state.play_card(card_index, board_position)
        
        elif self.action_type == ActionSpace.TAP_CARD:
            board_position = self.params.get('board_position', 0)
            return game_state.tap_card(board_position)
        
        elif self.action_type == ActionSpace.ATTACK:
            attacker_pos = self.params.get('attacker_pos', 0)
            target_pos = self.params.get('target_pos', 0)
            success, _ = game_state.attack(attacker_pos, target_pos)
            return success
        
        elif self.action_type == ActionSpace.DIRECT_ATTACK:
            attacker_pos = self.params.get('attacker_pos', 0)
            return game_state.direct_attack(attacker_pos)
        
        elif self.action_type == ActionSpace.SING:
            card_pos = self.params.get('card_pos', 0)
            success, _ = game_state.sing(card_pos)
            return success
        
        elif self.action_type == ActionSpace.GAIN_LORE:
            return game_state.gain_lore()
        
        elif self.action_type == ActionSpace.END_TURN:
            game_state.end_turn()
            return True
        
        return False
    
    def __str__(self):
        action_name = ActionSpace.get_action_name(self.action_type)
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{action_name}({params_str})"

# AI Agent implementation with Deep Q-Learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        if valid_actions is None or not valid_actions:
            # No valid actions, just end turn
            return ActionSpace.END_TURN
        
        if np.random.rand() <= self.epsilon:
            # Random action selection from valid actions
            return np.random.choice(valid_actions)
        
        act_values = self.model.predict(state, verbose=0)
        # Filter to only valid actions
        valid_acts = {action: act_values[0][action] for action in valid_actions}
        return max(valid_acts, key=valid_acts.get)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)


# Game environment for RL training
class CardGameEnv:
    def __init__(self, card_database: CardDatabase, deck_size: int = 30):
        self.card_database = card_database
        self.deck_size = deck_size
        self.game_state = None
        self.reset()
    
    def reset(self):
        # Create new decks
        player_cards = self.card_database.get_random_cards(self.deck_size)
        opponent_cards = self.card_database.get_random_cards(self.deck_size)
        
        player_deck = Deck(player_cards)
        opponent_deck = Deck(opponent_cards)
        
        # Initialize new game
        self.game_state = GameState(player_deck, opponent_deck)
        self.game_state.initialize_game()
        self.game_state.start_turn()
        
        # Return initial state
        return self._get_state_representation()
    
    def step(self, action):
        """Execute an action and return (next_state, reward, done, info)."""
        # Track state before action
        prev_lore = self.game_state.player.lore
        prev_opponent_cards = self.game_state.board.count_cards(1)
        prev_health = self.game_state.player.health
        prev_opponent_health = self.game_state.opponent.health
        
        # Create and execute the action
        game_action = self._create_action(action)
        success = game_action.execute(self.game_state)
        
        # Start opponent's turn if player ended turn
        if success and action == ActionSpace.END_TURN and self.game_state.current_player_id == 1:
            self.game_state.start_turn()
            self._opponent_turn()
        
        # Check win condition
        self.game_state.check_win_condition()
        
        # Calculate reward
        reward = self._calculate_reward(prev_lore, prev_opponent_cards, prev_health, prev_opponent_health, success)
        
        # Get next state
        next_state = self._get_state_representation()
        
        # Check if game is done
        done = self.game_state.game_over
        
        # Additional info
        info = {
            'success': success,
            'action': str(game_action),
            'reward': reward,
            'game_over': done,
            'winner': self.game_state.winner
        }
        
        return next_state, reward, done, info
    
    def _create_action(self, action):
        """Map the action to a GameAction with appropriate parameters."""
        if action == ActionSpace.PLAY_CARD:
            # Select a random valid card to play
            player = self.game_state.player
            valid_cards = [i for i in range(len(player.hand)) if player.can_play_card(i)]
            
            if not valid_cards:
                return GameAction(ActionSpace.END_TURN)
            
            card_index = random.choice(valid_cards)
            
            # Find an empty board position
            empty_positions = [i for i in range(MAX_BOARD_SIZE) 
                              if self.game_state.board.player_board[i] is None]
            
            if not empty_positions:
                return GameAction(ActionSpace.END_TURN)
            
            board_position = random.choice(empty_positions)
            
            return GameAction(ActionSpace.PLAY_CARD, {
                'card_index': card_index,
                'board_position': board_position
            })
        
        elif action == ActionSpace.TAP_CARD:
            # Find an untapped card to tap
            untapped = [i for i in range(MAX_BOARD_SIZE) 
                       if (self.game_state.board.player_board[i] is not None and 
                           not self.game_state.board.player_tapped[i])]
            
            if not untapped:
                return GameAction(ActionSpace.END_TURN)
            
            return GameAction(ActionSpace.TAP_CARD, {
                'board_position': random.choice(untapped)
            })
        
        elif action == ActionSpace.ATTACK:
            # Find an untapped card to attack with
            untapped = [i for i in range(MAX_BOARD_SIZE) 
                       if (self.game_state.board.player_board[i] is not None and 
                           not self.game_state.board.player_tapped[i])]
            
            if not untapped:
                return GameAction(ActionSpace.END_TURN)
            
            # Find an opponent card to attack
            targets = [i for i in range(MAX_BOARD_SIZE) 
                      if self.game_state.board.opponent_board[i] is not None]
            
            if not targets:
                return GameAction(ActionSpace.DIRECT_ATTACK, {
                    'attacker_pos': random.choice(untapped)
                })
            
            return GameAction(ActionSpace.ATTACK, {
                'attacker_pos': random.choice(untapped),
                'target_pos': random.choice(targets)
            })
        
        elif action == ActionSpace.DIRECT_ATTACK:
            # Find an untapped card to attack with
            untapped = [i for i in range(MAX_BOARD_SIZE) 
                       if (self.game_state.board.player_board[i] is not None and 
                           not self.game_state.board.player_tapped[i])]
            
            if not untapped:
                return GameAction(ActionSpace.END_TURN)
            
            return GameAction(ActionSpace.DIRECT_ATTACK, {
                'attacker_pos': random.choice(untapped)
            })
        
        elif action == ActionSpace.SING:
            # Find an untapped card with song effect
            untapped_songs = []
            for i in range(MAX_BOARD_SIZE):
                card = self.game_state.board.player_board[i]
                if (card is not None and 
                    not self.game_state.board.player_tapped[i] and 
                    card.song_effect):
                    untapped_songs.append(i)
            
            if not untapped_songs:
                return GameAction(ActionSpace.END_TURN)
            
            return GameAction(ActionSpace.SING, {
                'card_pos': random.choice(untapped_songs)
            })
        
        elif action == ActionSpace.GAIN_LORE:
            return GameAction(ActionSpace.GAIN_LORE)
        
        else:  # action == ActionSpace.END_TURN
            return GameAction(ActionSpace.END_TURN)
    
    def _opponent_turn(self):
        """Simple rule-based AI for opponent's turn."""
        while self.game_state.current_player_id == 1 and not self.game_state.game_over:
            # Play cards if possible
            opponent = self.game_state.opponent
            for card_index in range(len(opponent.hand)):
                if opponent.can_play_card(card_index):
                    # Find an empty board position
                    empty_positions = [i for i in range(MAX_BOARD_SIZE) 
                                      if self.game_state.board.opponent_board[i] is None]
                    
                    if empty_positions:
                        board_position = random.choice(empty_positions)
                        action = GameAction(ActionSpace.PLAY_CARD, {
                            'card_index': card_index,
                            'board_position': board_position
                        })
                        if action.execute(self.game_state):
                            continue
            
            # Try to attack
            for attacker_pos in range(MAX_BOARD_SIZE):
                if (self.game_state.board.opponent_board[attacker_pos] is not None and 
                        not self.game_state.board.opponent_tapped[attacker_pos]):
                    
                    # Direct attack if possible
                    action = GameAction(ActionSpace.DIRECT_ATTACK, {
                        'attacker_pos': attacker_pos
                    })
                    if action.execute(self.game_state):
                        continue
                    
                    # Otherwise attack a card
                    targets = [i for i in range(MAX_BOARD_SIZE) 
                              if self.game_state.board.player_board[i] is not None]
                    
                    if targets:
                        target_pos = random.choice(targets)
                        action = GameAction(ActionSpace.ATTACK, {
                            'attacker_pos': attacker_pos,
                            'target_pos': target_pos
                        })
                        if action.execute(self.game_state):
                            continue
            
            # Use song effects
            for card_pos in range(MAX_BOARD_SIZE):
                card = self.game_state.board.opponent_board[card_pos]
                if (card is not None and 
                    not self.game_state.board.opponent_tapped[card_pos] and 
                    card.song_effect):
                    
                    action = GameAction(ActionSpace.SING, {
                        'card_pos': card_pos
                    })
                    if action.execute(self.game_state):
                        continue
            
            # End turn
            action = GameAction(ActionSpace.END_TURN)
            action.execute(self.game_state)
        
        # Start player's turn
        if not self.game_state.game_over and self.game_state.current_player_id == 0:
            self.game_state.start_turn()
    
    def _get_state_representation(self):
        """Convert the game state to a neural network input vector."""
        # This is a simplified representation
        state = []
        
        # Player hand (up to MAX_HAND_SIZE cards, each with cost, attack, defense)
        hand_features = []
        for card in self.game_state.player.hand[:MAX_HAND_SIZE]:
            if card:
                hand_features.extend([card.cost, card.attack, card.defense])
            else:
                hand_features.extend([0, 0, 0])
        
        # Pad if less than MAX_HAND_SIZE cards
        padding = (MAX_HAND_SIZE - len(self.game_state.player.hand)) * 3
        if padding > 0:
            hand_features.extend([0] * padding)
        
        state.extend(hand_features)
        
        # Player board (card stats and tapped status)
        board_features = []
        for i in range(MAX_BOARD_SIZE):
            card = self.game_state.board.player_board[i]
            tapped = self.game_state.board.player_tapped[i]
            
            if card:
                board_features.extend([card.cost, card.attack, card.defense, 1 if tapped else 0])
            else:
                board_features.extend([0, 0, 0, 0])
        
        state.extend(board_features)
        
        # Opponent board
        opponent_board_features = []
        for i in range(MAX_BOARD_SIZE):
            card = self.game_state.board.opponent_board[i]
            tapped = self.game_state.board.opponent_tapped[i]
            
            if card:
                opponent_board_features.extend([card.cost, card.attack, card.defense, 1 if tapped else 0])
            else:
                opponent_board_features.extend([0, 0, 0, 0])
        
        state.extend(opponent_board_features)
        
        # Additional game state info
        state.extend([
            self.game_state.player.lore,
            self.game_state.player.health,
            len(self.game_state.player.inkwell),
            sum(1 for ink in self.game_state.player.inkwell if ink),
            self.game_state.opponent.lore,
            self.game_state.opponent.health,
            self.game_state.turn_number
        ])
        
        # Convert to numpy array and reshape for NN input
        return np.reshape(np.array(state), [1, len(state)])
    
    def _calculate_reward(self, prev_lore, prev_opponent_cards, prev_health, prev_opponent_health, success):
        """Calculate the reward for the last action."""
        reward = 0
        
        # Basic reward for successful actions
        if success:
            reward += 0.1
        
        # Reward for gaining lore
        lore_gained = self.game_state.player.lore - prev_lore
        if lore_gained > 0:
            reward += lore_gained * 5
        
        # Reward for eliminating enemy cards
        cards_eliminated = prev_opponent_cards - self.game_state.board.count_cards(1)
        if cards_eliminated > 0:
            reward += cards_eliminated * 10
        
        # Reward for causing damage to opponent
        damage_dealt = prev_opponent_health - self.game_state.opponent.health
        if damage_dealt > 0:
            reward += damage_dealt * 2
        
        # Penalty for taking damage
        damage_taken = prev_health - self.game_state.player.health
        if damage_taken > 0:
            reward -= damage_taken
        
        # Large reward/penalty for winning/losing
        if self.game_state.game_over:
            if self.game_state.winner == 0:  # Player wins
                reward += 100
            else:  # Opponent wins
                reward -= 50
        
        return reward
    
    def get_valid_actions(self):
        """Return a list of valid actions for the current state."""
        valid_actions = []
        
        # Always valid to end turn
        valid_actions.append(ActionSpace.END_TURN)
        
        # Can gain lore as an action
        valid_actions.append(ActionSpace.GAIN_LORE)
        
        # Check if any cards can be played
        if any(self.game_state.player.can_play_card(i) for i in range(len(self.game_state.player.hand))):
            valid_actions.append(ActionSpace.PLAY_CARD)
        
        # Check if there are any cards that can attack
        can_attack = False
        for i in range(MAX_BOARD_SIZE):
            if (self.game_state.board.player_board[i] is not None and 
                    not self.game_state.board.player_tapped[i]):
                can_attack = True
                break
        
        if can_attack:
            # If opponent has any cards, we can attack them
            if any(self.game_state.board.opponent_board[i] is not None for i in range(MAX_BOARD_SIZE)):
                valid_actions.append(ActionSpace.ATTACK)
            
            # Direct attack is possible if we have an untapped card
            valid_actions.append(ActionSpace.DIRECT_ATTACK)
        
        # Check if there are any cards with song effects
        for i in range(MAX_BOARD_SIZE):
            card = self.game_state.board.player_board[i]
            if (card is not None and 
                not self.game_state.board.player_tapped[i] and 
                card.song_effect):
                valid_actions.append(ActionSpace.SING)
                break
        
        # Tap can be done on any untapped card
        if can_attack:
            valid_actions.append(ActionSpace.TAP_CARD)
        
        return valid_actions


# Functions to run the simulation and training
def run_simulation(card_database, num_games=10):
    """Run a simple simulation with rule-based agents."""
    wins = {0: 0, 1: 0}
    
    for i in range(num_games):
        # Create game with random decks
        player_cards = card_database.get_random_cards(30)
        opponent_cards = card_database.get_random_cards(30)
        
        player_deck = Deck(player_cards)
        opponent_deck = Deck(opponent_cards)
        
        game = GameState(player_deck, opponent_deck)
        game.initialize_game()
        
        turn_limit = 100  # Prevent infinite games
        current_turn = 0
        
        # Main game loop
        while not game.game_over and current_turn < turn_limit:
            current_turn += 1
            
            # Start turn
            game.start_turn()
            
            # Simple rule-based AI for both players
            player_id = game.current_player_id
            current_player = game.player if player_id == 0 else game.opponent
            
            # Play phase - play as many cards as possible
            for card_index in range(len(current_player.hand)):
                if current_player.can_play_card(card_index):
                    # Find an empty board position
                    empty_positions = []
                    board = game.board.player_board if player_id == 0 else game.board.opponent_board
                    
                    for i in range(MAX_BOARD_SIZE):
                        if board[i] is None:
                            empty_positions.append(i)
                    
                    if empty_positions:
                        board_position = random.choice(empty_positions)
                        game.play_card(card_index, board_position)
            
            # Attack phase - attack with all possible cards
            for attacker_pos in range(MAX_BOARD_SIZE):
                attacker_board = game.board.player_board if player_id == 0 else game.board.opponent_board
                attacker_tapped = game.board.player_tapped if player_id == 0 else game.board.opponent_tapped
                
                if attacker_board[attacker_pos] is not None and not attacker_tapped[attacker_pos]:
                    # Try direct attack first
                    if game.direct_attack(attacker_pos):
                        continue
                    
                    # Otherwise try to attack a card
                    target_board = game.board.opponent_board if player_id == 0 else game.board.player_board
                    targets = [i for i in range(MAX_BOARD_SIZE) if target_board[i] is not None]
                    
                    if targets:
                        target_pos = random.choice(targets)
                        game.attack(attacker_pos, target_pos)
            
            # Sing phase - use any available song effects
            for card_pos in range(MAX_BOARD_SIZE):
                board = game.board.player_board if player_id == 0 else game.board.opponent_board
                tapped = game.board.player_tapped if player_id == 0 else game.board.opponent_tapped
                
                card = board[card_pos]
                if card is not None and not tapped[card_pos] and card.song_effect:
                    game.sing(card_pos)
            
            # End turn
            game.end_turn()
        
        # Track winner
        if game.winner is not None:
            wins[game.winner] += 1
        
        logger.info(f"Game {i+1} completed in {current_turn} turns. Winner: {game.winner}")
    
    logger.info(f"Simulation results - Player wins: {wins[0]}, Opponent wins: {wins[1]}")
    return wins


def train_dqn_agent(card_database, episodes=200, batch_size=32, checkpoint_freq=50):
    """Train a DQN agent to play the card game."""
    env = CardGameEnv(card_database)
    
    # Define state and action sizes
    state_size = len(env._get_state_representation()[0])
    action_size = 7  # Number of action types
    
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    rewards_history = []
    wins = 0
    
    for episode in tqdm(range(episodes), desc="Training DQN Agent"):
        print(f"\n🔹 Starting Episode {episode+1}/{episodes}")  # Debugging
        
        # Reset environment
        state = env.reset()
        #print(f"Initial State: {state}")  # Debugging
        
        total_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            #print(f"Valid Actions: {valid_actions}")  # Debugging
            
            # Select and perform action
            action = agent.act(state, valid_actions)
            #print(f"Agent Chose Action: {action}")  # Debugging
            
            next_state, reward, done, info = env.step(action)
            #print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")  # Debugging
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            #print(f"Stored Experience: {state, action, reward, next_state, done}")  # Debugging
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Learn from experience
            agent.replay(batch_size)
            #print(f"Training step completed for episode {episode+1}")  # Debugging
            
            # Update target network periodically
            if episode % 10 == 0:
                agent.update_target_model()
        
        # Track metrics
        rewards_history.append(total_reward)
        if env.game_state.winner == 0:
            wins += 1
        
        # Save model periodically
        if episode > 0 and episode % checkpoint_freq == 0:
            print(f"🔄 Saving Model at Episode {episode}")  # Debugging
            agent.save(f"dqn_agent_episode_{episode}.weights.h5")
            
            # Calculate win rate over last checkpoints
            win_rate = wins / checkpoint_freq
            wins = 0
            
            logger.info(f"Episode {episode}/{episodes} - Win rate: {win_rate:.2f}")
            
            # Plot rewards
            if len(rewards_history) > 10:
                plt.figure(figsize=(10, 5))
                plt.plot(rewards_history)
                plt.title(f"DQN Training Rewards - Episode {episode}")
                plt.xlabel("Episode")
                plt.ylabel("Total Reward")
                plt.savefig(f"rewards_episode_{episode}.png")
                plt.close()
    
    # Save final model
    agent.save("dqn_agent_final.weights.h5")
    
    logger.info("Training completed!")
    return agent, rewards_history



def evaluate_agent(agent, card_database, num_games=100):
    env = CardGameEnv(card_database)
    
    wins = 0
    rewards = []
    
    for i in tqdm(range(num_games), desc="Evaluating Agent"):
        state = env.reset()
        done = False
        total_reward = 0
        turn_count = 0
        max_turns = 100  # Add a turn limit
        
        while not done and turn_count < max_turns:
            turn_count += 1
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action without exploration
            agent.epsilon = 0.0
            action = agent.act(state, valid_actions)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update
            state = next_state
            total_reward += reward
        
        # If the game didn't end naturally, force it to end
        if not done:
            #logger.warning(f"Game {i+1} exceeded turn limit ({max_turns} turns). Forcing end.")
            if env.game_state.player.health > env.game_state.opponent.health:
                env.game_state.winner = 0
            else:
                env.game_state.winner = 1
            done = True
        
        rewards.append(total_reward)
        if env.game_state.winner == 0:
            wins += 1
    
    win_rate = wins / num_games
    avg_reward = sum(rewards) / num_games
    
    logger.info(f"Evaluation results:")
    logger.info(f"Win rate: {win_rate:.2f}")
    logger.info(f"Average reward: {avg_reward:.2f}")
    
    return win_rate, avg_reward


def main():
    """Main function to run the card game simulation and training."""
    # Create card database
    card_db = CardDatabase("cards.json")
    
    # Run initial simulation to verify game mechanics
    logger.info("Running initial simulation to verify game mechanics...")
    run_simulation(card_db, num_games=10)
    
    # Train DQN agent
    logger.info("Training DQN agent...")
    agent, rewards = train_dqn_agent(card_db, episodes=200, batch_size=32)
    
    # Evaluate agent
    logger.info("Evaluating trained agent...")
    evaluate_agent(agent, card_db, num_games=100)
    
    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("training_rewards.png")
    
    logger.info("Phase 1 completed!")


if __name__ == "__main__":
    main()