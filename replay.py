
from floodsmart_env_playwright_vision import FloodSmartPlaywrightVisionEnv
from cnn_dqn_tensorzero_agent import VisionDQNAgent
from PIL import Image
from utils import save_gif_from_frames

def replay_episode(model_path="models/qagent_cnn.pt", save_path="logs/episode_replay.gif"):
    env = FloodSmartPlaywrightVisionEnv()
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n
    agent = VisionDQNAgent(env, state_shape, action_dim)

    agent.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.model.eval()

    frames = []
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < 50:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        frame = Image.fromarray(state[:, :, :3])
        frames.append(frame)
        state = next_state
        steps += 1

    save_gif_from_frames(frames, save_path)
    env.close()
