
# 🤖 QAgent – Vision-Based AI Test Navigation Agent

QAgent is an AI-powered test automation agent built using reinforcement learning, Playwright, and computer vision. It simulates a real user navigating a web app — learning UI patterns and making intelligent decisions.

---

## 📦 Features

- 🧠 Trained with Deep Q-Learning using vision-based screenshots + DOM context
- 🌐 Automates real browsers via Playwright
- 🎞️ Captures GIF replays of test episodes
- 📊 Logs reward trends and training progress
- 🖥️ Interactive Streamlit UI for demo
- 🧪 Google Colab notebook for cloud-based testing

---

## 🛠️ Local Setup Instructions

### 1. Clone or Unzip the Project

```bash
git clone https://github.com/yourname/qagent.git
cd qagent
# OR just unzip QAgent_Final.zip and cd into it
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv qagent-env
source qagent-env/bin/activate  # Windows: qagent-env\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
playwright install
```

---

## 🚀 Run Streamlit App

```bash
streamlit run app.py
```

### What You Can Do:
- ▶️ Run a test episode with the trained agent
- 🎞️ View latest GIF replays
- 📊 Load trained model from `models/qagent_cnn.pt`

---

## 🧠 Train the Agent

```bash
python train_qagent.py
```

This will:
- Train the model
- Save it to `models/qagent_cnn.pt`
- Log episode reward trends
- Save GIFs to `logs/`

---

## 🔁 Replay Saved Episodes

```bash
python replay.py
```

This loads your model and replays an episode, saving a new GIF to `logs/episode_replay.gif`.

---

## 📔 Run in Google Colab

Open `QAgent_colab_demo.ipynb` in Google Colab:
- Upload your `models/qagent_cnn.pt` file
- Run one test episode
- Automatically generate and display a replay GIF

---

## 📁 Folder Structure

```
QAgent/
├── app.py                 # Streamlit demo UI
├── train_qagent.py        # Training loop
├── replay.py              # Replay agent and save GIF
├── utils.py               # GIF saving + reward plotting
├── cnn_dqn_tensorzero_agent.py
├── floodsmart_env_playwright_vision.py
├── QAgent_colab_demo.ipynb
├── requirements.txt
├── models/
│   └── qagent_cnn.pt      # Trained agent (if exists)
└── logs/
    └── episode_001.gif    # Episode replays
```

---

## 💡 Tips

- Modify the environment in `floodsmart_env_playwright_vision.py` to explore new sites or workflows
- Use `plot_rewards()` from `utils.py` to generate graphs after training
- Customize the Streamlit UI in `app.py` to add new buttons, inputs, or insights

---

Built with ❤️ for testers, developers, and AI explorers.
