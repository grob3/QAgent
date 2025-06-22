
import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image

def save_gif_from_frames(frames, path, duration=0.3):
    """
    Save a list of PIL images as a gif.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, duration=duration)

def plot_rewards(reward_history, save_path="logs/reward_plot.png"):
    """
    Plot and save episode rewards.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("QAgent Training Reward History")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
