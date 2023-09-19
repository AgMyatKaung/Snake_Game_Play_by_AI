import streamlit as st
import torch
from game import SnakeGameAI, Direction, Point
from agent import Agent  # Import your Agent class from agent.py
import matplotlib.pyplot as plt
import time
import subprocess  # Import subprocess module

# Function to run the game loop
def run_game(agent, game):
    scores = []
    highest_score = 0

    chart = st.line_chart(scores)  # Initialize the live chart

    while True:
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

# Set page title and description
st.set_page_config(page_title="Snake Game AI", page_icon="🐍")
st.title("Snake Game AI")

# Create a two-column layout
col1, col2 = st.columns([2, 1])

# Column 1: Snake Game AI
with col1:
    st.subheader("Game Display")
    
    # Create or load the agent and game instances
    agent = Agent()
    agent.load_model()
    game = SnakeGameAI()

    # Display game controls
    st.markdown("## Game Controls")
    st.text("Press 'Ctrl+C' to stop the game.")

    # Check if "Train AI" button is clicked
    if st.button("Train AI"):
        st.text("AI training in progress...")
        
        # Use subprocess to run the game in the console
        subprocess.Popen(["python", "your_game_script.py"])

# Column 2: Buttons
with col2:
    st.subheader("Actions")
    
    # Display buttons for actions
    if st.button("End Train"):
        st.text("Game stopped.")

    # You can remove the "Total Games Played" and "Highest Score" sections
    # st.markdown("## Training Progress")
    # st.text(f"Total Games Played: {agent.n_games}")
    # st.text(f"Highest Score: {agent.highest_score}")

# Add a divider for better separation
st.markdown("---")

# Display live chart in the main section
st.subheader("Live Score Chart")
