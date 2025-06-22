
import streamlit as st
import os
import numpy as np
import torch
from floodsmart_env_playwright_vision import FloodSmartPlaywrightVisionEnv
from cnn_dqn_tensorzero_agent import VisionDQNAgent
from PIL import Image
import glob

st.set_page_config(page_title="QAgent UI", layout="wide")

st.title("🤖 QAgent – Vision-Based Test Navigation Agent")

# Load model
MODEL_PATH = "models/qagent_cnn.pt"

@st.cache_resource
def load_agent():
    env = FloodSmartPlaywrightVisionEnv()
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n
    agent = VisionDQNAgent(env, state_shape, action_dim)
    if os.path.exists(MODEL_PATH):
        agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        agent.model.eval()
    return agent, env

agent, env = load_agent()

# Run evaluation
if st.button("▶️ Run One Test Episode"):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    stframe = st.empty()

    while not done and step_count < 50:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        img = Image.fromarray(state[:, :, :3])
        stframe.image(img, caption=f"Step {step_count}, Action {action}, Reward {reward}", use_column_width=True)
        step_count += 1

    env.close()
    st.success(f"Episode finished in {step_count} steps with reward {total_reward}")

# Replay GIFs
st.subheader("🎞️ Episode Replays")
gif_files = sorted(glob.glob("logs/*.gif"))
if gif_files:
    for gif in gif_files[-5:][::-1]:
        st.image(gif, caption=os.path.basename(gif), use_column_width=True)
else:
    st.info("No GIFs found in logs/ yet. Train or run an episode first.")
