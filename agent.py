import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

SAVE_EVERY = 5  # Save the model every 5 games

TRAINING_FOLDER = 'Snake_Game_Play_by_AI'

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.highest_score = 0


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, file_name='snake_model.pth'):
        model_state = {
        'model_state_dict': self.model.state_dict(),
        'highest_score': self.highest_score  # Include highest score
    }
        torch.save(self.model.state_dict(), os.path.join(TRAINING_FOLDER, file_name))

    def load_model(self, file_name='snake_model.pth'):
        file_path = os.path.join(TRAINING_FOLDER, file_name)
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint.get('model_state_dict', self.model.state_dict()))
            self.model.eval()
            self.highest_score = checkpoint.get('highest_score', 0)  # Retrieve highest score from checkpoint
            print("Model loaded successfully.")
        else:
            print(f"No model file found at {file_path}. Training from scratch.")
    
    def load_previous_training(self):
        # Load the previous training data (memory and model) if available
        memory_file = os.path.join(TRAINING_FOLDER, 'memory.npy')
        model_file = os.path.join(TRAINING_FOLDER, 'snake_model.pth')

        if os.path.exists(memory_file) and os.path.exists(model_file):
            print("Loading previous training data...")
            
            # Load memory
            self.memory = deque(np.load(memory_file, allow_pickle=True), maxlen=MAX_MEMORY)
            
            # Load model
            checkpoint = torch.load(model_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load highest score
            self.highest_score = checkpoint.get('highest_score', 0)
            
            print("Previous training data loaded successfully.")

    def save_current_training(self):
        # Save the current memory and model
        memory_file = os.path.join(TRAINING_FOLDER, 'memory.npy')
        model_file = os.path.join(TRAINING_FOLDER, 'snake_model.pth')

        np.save(memory_file, np.array(self.memory))
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'highest_score': self.highest_score,
        }, model_file)
        print("Current training data saved.")




def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    agent.load_previous_training()

    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score, highest_score = game.play_step(final_move)  # Include highest_score
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save_model()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Highest Score:', highest_score)  # Print highest score
            # Update the highest score in the agent (if needed)
            if highest_score > agent.highest_score:
                agent.highest_score = highest_score

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if agent.n_games % SAVE_EVERY == 0:
                agent.save_current_training()

if __name__ == '__main__':
    agent = Agent()
    if not agent.load_model():
        train()
