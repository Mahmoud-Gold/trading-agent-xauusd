"""Deep Q-Network Agent for Trading."""

import numpy as np
import tensorflow as tf
from collections import deque
from loguru import logger

from config.settings import (
    STATE_SIZE, ACTION_SIZE, MEMORY_SIZE, BATCH_SIZE, GAMMA,
    LEARNING_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    UPDATE_TARGET_FREQ, NOISE_ENABLED, NOISE_LEVEL, NOISE_DECAY
)


class DQNAgent:
    """Deep Q-Network Agent for autonomous trading."""
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        """Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions (Buy/Hold/Sell)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Learning parameters
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        # Noise parameters
        self.noise_enabled = NOISE_ENABLED
        self.noise_level = NOISE_LEVEL
        self.noise_decay = NOISE_DECAY
        
        # Experience replay
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batch_size = BATCH_SIZE
        
        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training tracking
        self.steps = 0
        self.update_target_freq = UPDATE_TARGET_FREQ
        
        logger.info(f"DQN Agent initialized - State: {state_size}, Actions: {action_size}")
    
    def _build_model(self):
        """Build neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update target network weights."""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated")
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (use exploration)
            
        Returns:
            Action index (0=Buy, 1=Hold, 2=Sell)
        """
        if training and np.random.random() <= self.epsilon:
            # Exploration: random action
            action = np.random.choice(self.action_size)
        else:
            # Exploitation: best action from model
            state = np.reshape(state, [1, self.state_size])
            q_values = self.model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
        
        # Apply noise if enabled
        if training and self.noise_enabled and np.random.random() < self.noise_level:
            action = np.random.choice(self.action_size)
        
        return action
    
    def replay(self, batch_size=None):
        """Experience replay training.
        
        Args:
            batch_size: Size of batch to train on
            
        Returns:
            Loss value
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return None
        
        # Sample random batch from memory
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Decay noise
        if self.noise_enabled:
            self.noise_level *= self.noise_decay
        
        self.steps += 1
        
        # Update target network periodically
        if self.steps % self.update_target_freq == 0:
            self.update_target_model()
        
        return loss
    
    def save(self, filepath):
        """Save model to file."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        self.target_model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_memory_size(self):
        """Get current memory size."""
        return len(self.memory)
