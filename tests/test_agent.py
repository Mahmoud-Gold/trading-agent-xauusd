"""Unit tests for DQN Agent."""

import pytest
import numpy as np
from src.agent.dqn_agent import DQNAgent
from config.settings import STATE_SIZE, ACTION_SIZE


class TestDQNAgent:
    """Test DQN Agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        assert self.agent.state_size == STATE_SIZE
        assert self.agent.action_size == ACTION_SIZE
        assert self.agent.epsilon == 1.0
        assert len(self.agent.memory) == 0
    
    def test_agent_act(self):
        """Test agent action selection."""
        state = np.random.randn(STATE_SIZE)
        action = self.agent.act(state, training=True)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < ACTION_SIZE
    
    def test_agent_remember(self):
        """Test experience replay memory."""
        state = np.random.randn(STATE_SIZE)
        action = 0
        reward = 1.0
        next_state = np.random.randn(STATE_SIZE)
        done = False
        
        self.agent.remember(state, action, reward, next_state, done)
        
        assert len(self.agent.memory) == 1
    
    def test_agent_replay(self):
        """Test experience replay training."""
        # Fill memory
        for _ in range(50):
            state = np.random.randn(STATE_SIZE)
            action = np.random.randint(ACTION_SIZE)
            reward = np.random.randn()
            next_state = np.random.randn(STATE_SIZE)
            done = False
            
            self.agent.remember(state, action, reward, next_state, done)
        
        loss = self.agent.replay(batch_size=32)
        
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_agent_save_load(self, tmp_path):
        """Test model save and load."""
        filepath = tmp_path / "test_model.h5"
        
        # Save model
        self.agent.save(str(filepath))
        assert filepath.exists()
        
        # Load model
        new_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        new_agent.load(str(filepath))
        
        assert new_agent.model is not None
