import os
import subprocess

import gymnasium as gym
import numpy as np
import torch

from trainer import TrainIrlGrid
from trainer import expert_policy as default_policy

RL_ZOO_REPO = "https://github.com/DLR-RM/rl-baselines3-zoo.git"
RL_ZOO_DIR = "rl-baselines3-zoo"


def ensure_rl_zoo_cloned():
    """RL-Baselines3-Zoo をローカルにクローンし、サブモジュールも初期化／更新する"""
    if not os.path.isdir(RL_ZOO_DIR):
        print(f"Cloning RL-Baselines3-Zoo into ./{RL_ZOO_DIR} …")
        subprocess.run(["git", "clone", "--depth", "1", RL_ZOO_REPO], check=True)
        subprocess.run(
            ["git", "-C", RL_ZOO_DIR, "submodule", "update", "--init", "--recursive"], check=True
        )
    else:
        print(f"{RL_ZOO_DIR} already exists, pulling latest changes…")
        subprocess.run(["git", "-C", RL_ZOO_DIR, "pull"], check=True)
        subprocess.run(
            ["git", "-C", RL_ZOO_DIR, "submodule", "update", "--recursive", "--remote"], check=True
        )


def load_expert_model_from_zoo(algo: str, env_name: str):
    """
    RL-Baselines3-Zoo の該当ディレクトリから学習済みモデルをロード
    例: rl-trained-agents/dqn/CartPole-v1/model.zip
    """
    model_subdir = algo.lower()
    model_path = os.path.join(RL_ZOO_DIR, "rl-trained-agents", model_subdir, env_name, "model.zip")
    if not os.path.isfile(model_path):
        print(f"Warning: Zoo model not found at {model_path!r}, falling back to default policy.")
        return None
    env = gym.make(env_name)
    if algo.upper() == "DQN":
        from stable_baselines3 import DQN

        return DQN.load(model_path, env=env)
    from stable_baselines3 import PPO

    return PPO.load(model_path, env=env)


def validate_irl_with_open_model(
    env_name: str,
    expert_model_path: str = None,
    algo: str = "DQN",
    num_epochs: int = 200,
    num_eval: int = 500,
    max_steps: int = 50,
    use_zoo: bool = True,
):
    """
    1) Zoo からもしくはローカル指定から SB3 モデルをロード
    2) TrainIrlGrid で IRL モデルを学習
    3) 専門家 vs ランダムの累計推定報酬を比較して成功判定
    """
    # 1. SB3 モデルのロード
    sb3_model = None
    if use_zoo:
        ensure_rl_zoo_cloned()
        sb3_model = load_expert_model_from_zoo(algo, env_name)
    elif expert_model_path:
        env = gym.make(env_name)
        if algo.upper() == "DQN":
            from stable_baselines3 import DQN

            sb3_model = DQN.load(expert_model_path, env=env)
        else:
            from stable_baselines3 import PPO

            sb3_model = PPO.load(expert_model_path, env=env)

    # 2. 専門家ポリシーの設定
    if sb3_model:
        expert_policy_fn = lambda s: sb3_model.predict(s, deterministic=True)[0]
    else:
        print("Using default rule-based expert policy.")
        expert_policy_fn = default_policy

    # 3. IRL トレーニング
    trainer = TrainIrlGrid(
        env_name=env_name, expert_policy=expert_policy_fn, num_epoch=num_epochs, state_space=None
    )
    irl_model = trainer.train()

    # 4. 評価（専門家 vs ランダム）
    env = gym.make(env_name)
    expert_scores, random_scores = [], []

    for _ in range(num_eval):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total = 0.0
        for _ in range(max_steps):
            action = expert_policy_fn(state)
            step = env.step(action)
            if len(step) == 5:
                next_state, _, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, _, done, _ = step
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            # 報酬推定: 観測を擬似画像化して reward_net へ
            obs = (
                torch.tensor(state, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(irl_model.device)
            )
            total += irl_model.reward_net(obs).item()
            state = next_state
            if done:
                break
        expert_scores.append(total)

    for _ in range(num_eval):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total = 0.0
        for _ in range(max_steps):
            action = env.action_space.sample()
            step = env.step(action)
            if len(step) == 5:
                next_state, _, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, _, done, _ = step
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            obs = (
                torch.tensor(state, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(irl_model.device)
            )
            total += irl_model.reward_net(obs).item()
            state = next_state
            if done:
                break
        random_scores.append(total)

    expert_mean = np.mean(expert_scores)
    random_mean = np.mean(random_scores)
    print(f"[Validation] Expert mean reward: {expert_mean:.4f}")
    print(f"[Validation] Random mean reward: {random_mean:.4f}")
    print("[Validation] SUCCESS" if expert_mean > random_mean else "[Validation] FAILURE")

    return {
        "expert_mean": expert_mean,
        "random_mean": random_mean,
        "success": expert_mean > random_mean,
    }


if __name__ == "__main__":
    results = validate_irl_with_open_model(
        env_name="CartPole-v1",
        algo="DQN",
        num_epochs=100,
        num_eval=200,
        max_steps=200,
        use_zoo=True,
    )
    print(results)
