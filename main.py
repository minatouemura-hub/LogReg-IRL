import argparse
import os

import gymnasium as gym
from stable_baselines3 import DQN, PPO

from trainer import expert_policy  # ルールベースポリシー
from validation.valid.valid import validate_irl_with_open_model


def get_args():
    parser = argparse.ArgumentParser(description="Train expert and run IRL inference")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument(
        "--algo", type=str, choices=["DQN", "PPO"], default="DQN", help="Expert algorithm"
    )
    parser.add_argument(
        "--train_timesteps", type=int, default=100000, help="Timesteps for expert training"
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of IRL training epochs")
    parser.add_argument("--num_eval", type=int, default=500, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--expert_model_path",
        type=str,
        default=None,
        help="Path to local pre-trained SB3 model (zip). If not set, will train locally.",
    )
    return parser.parse_args()


def train_expert(env_name, algo, timesteps, save_path):
    env = gym.make(env_name)
    if algo == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
    else:
        model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    # ディレクトリ部分を取得し、存在する場合のみ作成
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    model.save(save_path)
    return save_path


def main():
    args = get_args()

    # Expert model: train locally if no path given
    if args.expert_model_path is None:
        default_path = f"expert_{args.algo}_{args.env_name}.zip"
        print(f"Training expert model locally: {default_path}")
        expert_path = train_expert(args.env_name, args.algo, args.train_timesteps, default_path)
    else:
        expert_path = args.expert_model_path

    # IRL inference using the trained expert
    results = validate_irl_with_open_model(
        env_name=args.env_name,
        expert_model_path=expert_path,
        algo=args.algo,
        num_epochs=args.num_epochs,
        num_eval=args.num_eval,
        max_steps=args.max_steps,
        use_zoo=False,  # always use local model
    )

    print("=== Validation Results ===")
    print(f"Expert Mean Reward : {results['expert_mean']:.4f}")
    print(f"Random Mean Reward : {results['random_mean']:.4f}")
    print("Success" if results["success"] else "Failure")


if __name__ == "__main__":
    main()
