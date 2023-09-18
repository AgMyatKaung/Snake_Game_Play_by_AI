import torch
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import numpy as np
import random
from agent import Agent  # Import the Agent class from agent.py

def play_game(model, agent):
    highest_score = 0  # Initialize the highest score
    
    while True:
        # Initialize the game
        game = SnakeGameAI()

        while True:
            # Get the current state
            state = agent.get_state(game)

            # Choose the best action using the pre-trained model
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = model(state_tensor)
            action = torch.argmax(prediction).item()

            # Translate the action into game commands
            if action == 0:
                move = Direction.LEFT
            elif action == 1:
                move = Direction.RIGHT
            else:
                # Choose a valid direction (UP or DOWN) when action is not 0 or 1
                move = Direction.UP

            # Play one step and check if the game is over
            _, done, _ = game.play_step(move)

            if done:
                current_score = game.score
                print(f"Game over! Current Score: {current_score}, Highest Score: {highest_score}")
                
                # Update the highest score if necessary
                if current_score > highest_score:
                    highest_score = current_score

                break

        # Automatically restart the game for continuous play
        print("Restarting the game...")
        game.reset()

if __name__ == '__main__':
    # Load the pre-trained model
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load("AI/AI_finalProject/snake_model.pth"))
    model.eval()

    # Create an instance of the Agent class
    agent = Agent()

    # Play the game using the loaded model and the agent
    play_game(model, agent)
