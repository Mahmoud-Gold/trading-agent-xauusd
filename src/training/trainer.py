"""Training pipeline for DQN Agent."""

import numpy as np
from loguru import logger
from tqdm import tqdm

from config.settings import TOTAL_STEPS, LOG_INTERVAL, TRAINING_EPISODES, EPISODE_STEPS


class Trainer:
    """Train DQN Agent."""
    
    def __init__(self, agent, environment):
        """Initialize trainer.
        
        Args:
            agent: DQN Agent instance
            environment: Trading environment
        """
        self.agent = agent
        self.environment = environment
        self.training_history = []
        
        logger.info("Trainer initialized")
    
    def train_episodes(self, episodes=TRAINING_EPISODES):
        """Train agent for specified episodes.
        
        Args:
            episodes: Number of episodes to train
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in tqdm(range(episodes), desc="Training Episodes"):
            state = self.environment.reset()
            episode_reward = 0
            episode_loss = []
            
            for step in range(EPISODE_STEPS):
                # Agent selects action
                action = self.agent.act(state, training=True)
                
                # Environment executes action
                next_state, reward, done, info = self.environment.step(action)
                
                # Agent learns
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.replay()
                
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Log episode statistics
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            
            if (episode + 1) % LOG_INTERVAL == 0:
                metrics = self.environment.get_metrics()
                logger.info(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Balance: ${metrics['final_balance']:.2f}"
                )
            
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon,
                'metrics': self.environment.get_metrics()
            })
        
        logger.info("Training completed")
    
    def train_steps(self, total_steps=TOTAL_STEPS):
        """Train agent for specified steps (5M steps).
        
        Args:
            total_steps: Total training steps
        """
        logger.info(f"Starting training for {total_steps:,} steps")
        
        state = self.environment.reset()
        step_count = 0
        episode = 0
        episode_reward = 0
        
        with tqdm(total=total_steps, desc="Training Steps") as pbar:
            while step_count < total_steps:
                # Agent selects action
                action = self.agent.act(state, training=True)
                
                # Environment executes action
                next_state, reward, done, info = self.environment.step(action)
                
                # Agent learns
                self.agent.remember(action, reward, next_state, done)
                loss = self.agent.replay()
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                # Log progress
                if step_count % LOG_INTERVAL == 0:
                    logger.debug(
                        f"Step {step_count:,} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Epsilon: {self.agent.epsilon:.4f}"
                    )
                
                pbar.update(1)
                
                if done:
                    episode += 1
                    logger.info(
                        f"Episode {episode} - "
                        f"Steps: {step_count:,} - "
                        f"Reward: {episode_reward:.2f}"
                    )
                    state = self.environment.reset()
                    episode_reward = 0
        
        logger.info(f"Training completed - Total steps: {step_count:,}, Episodes: {episode}")
    
    def get_training_history(self):
        """Get training history."""
        return self.training_history
