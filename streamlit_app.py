import streamlit as st
from game import SnakeGameAI, Direction, Point
from agent import Agent  # Import your Agent class from agent.py
import matplotlib.pyplot as plt
import time

# Function to run the game loop
def run_game(agent, game, stop_flag):
    scores = []
    highest_score = 0

    chart = st.line_chart(scores)  # Initialize the live chart

    while not stop_flag[0]:  # Use a flag to control the game loop
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score, highest_score = game.play_step(final_move)

        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > highest_score:
                highest_score = score

            scores.append(score)

            # Update the live chart
            chart.line_chart(scores)

            st.write(f"Game {agent.n_games}: Score {score}, Highest Score: {highest_score}")

        # Add a sleep to control the speed of the game (adjust as needed)
        time.sleep(0.01)  # Set a very low sleep duration to speed up the game

# Create buttons for training and stopping the game
st.title("Snake Game AI")
st.text("Press 'Ctrl+C' to stop the game.")
train_btn = st.button("Train AI")
stop_btn = st.button("Stop")

# Initialize the stop flag outside of the if block
stop_flag = [False]

if train_btn:
    if not stop_flag[0]:  # Check if the game is not already running
        st.text("AI training in progress...")

        # Create or load the agent and game instances
        agent = Agent()
        agent.load_model()
        game = SnakeGameAI()

        # Run the game loop
        run_game(agent, game, stop_flag)
    else:
        st.text("Resuming AI training...")

if stop_btn:
    st.text("Stopping AI training...")
    stop_flag[0] = True  # Set the flag to stop the game loop
