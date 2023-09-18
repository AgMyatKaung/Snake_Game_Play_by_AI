import pygame
import torch
from game import SnakeGameAI  # Import your game class
from agent import Agent  # Import your Agent class
from helper import plot
import os

Initialize Pygame
pygame.init()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
SAVE_EVERY = 5  # Save the model every 5 games

Folder where the trained model is saved
TRAINING_FOLDER = "training_data"

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)

def load_and_run():
    game = SnakeGameAI()

Create an agent instance
    agent = Agent()

    # Load the trained model
    model_file = os.path.join(TRAINING_FOLDER, "snake_model.pth")
    if os.path.exists(model_file):
        agent.load_model(model_file)
    else:
        print(f"No model file found at {model_file}. Training from scratch.")
        return

    while True:
        state = agent.get_state(game)
        action = agent.getaction(state)
        , done, _ = game.play_step(action)

        if done:
            game.reset()

if name == 'main':
    load_and_run()