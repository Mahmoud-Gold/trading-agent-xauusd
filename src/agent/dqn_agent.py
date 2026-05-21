"""Deep Q-Network Agent for Trading - Optimized Version."""

import numpy as np
import tensorflow as tf
from collections import deque
from loguru import logger
import os

from config.settings import (
    STATE_SIZE, ACTION_SIZE, MEMORY_SIZE, BATCH_SIZE, GAMMA,
    LEARNING_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    UPDATE_TARGET_FREQ, NOISE_ENABLED, NOISE_LEVEL, NOISE_DECAY
)


class DQNAgent:
    """Deep Q-Network Agent for autonomous trading - Optimized."""
    
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
        """Build optimized neural network model."""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(self.state_size,)),
            
            # Hidden layers - أصغر وأسرع
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber()  # أفضل من MSE للـ RL
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
        """Experience replay training - Optimized.
        
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
        
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch])
        
        # Predict Q-values - batch processing أسرع
        with tf.GradientTape() as tape:
            targets = self.model(states, training=True)
            next_q_values = self.target_model(next_states, training=False)
            
            # حساب targets
            target_values = targets.numpy().copy()
            for i in range(batch_size):
                if dones[i]:
                    target_values[i][actions[i]] = rewards[i]
                else:
                    target_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i].numpy())
            
            # حساب loss
            loss = tf.keras.losses.Huber()(targets, target_values)
        
        # تطبيق gradient
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
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
        
        return float(loss.numpy())
    
    def save(self, filepath):
        """Save model to file - بدون مشاكل serialization."""
        # إنشاء المجلد إذا ما كانش موجود
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # حفظ الأوزان بدل الـ model كامل
        self.model.save_weights(filepath.replace('.h5', '_weights.h5'))
        
        # حفظ config
        import json
        config = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': float(self.epsilon)
        }
        with open(filepath.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        try:
            # تحميل الأوزان
            weights_file = filepath.replace('.h5', '_weights.h5')
            if os.path.exists(weights_file):
                self.model.load_weights(weights_file)
                self.target_model.load_weights(weights_file)
                logger.info(f"Model loaded from {weights_file}")
            else:
                logger.warning(f"Weights file not found: {weights_file}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    
    def get_memory_size(self):
        """Get current memory size."""
        return len(self.memory)
