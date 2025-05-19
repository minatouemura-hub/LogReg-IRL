import argparse


def get_args():
    """
    コマンドライン引数をパースして返します。
    --env_name: Gym 環境名
    --expert_model_path: Stable Baselines3 モデルの読み込みパス
    --algo: 専門家モデルのアルゴリズム (DQN または PPO)
    --num_epochs: IRL モデル学習のエポック数
    --num_eval: 評価時のサンプル数
    --max_steps: 1エピソードあたりの最大ステップ数
    """
    parser = argparse.ArgumentParser(description="IRL with Open Expert Model on GridWorld")
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Gym environment name (default: GridWorld-v0)",
    )
    parser.add_argument(
        "--expert_model_path",
        type=str,
        default=None,
        help="Path to the pre-trained SB3 expert model",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["DQN", "PPO"],
        default="DQN",
        help="SB3 expert algorithm (DQN or PPO, default: DQN)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs for IRL training (default: 200)",
    )
    parser.add_argument(
        "--num_eval", type=int, default=500, help="Number of episodes for evaluation (default: 500)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps per episode (default: 50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
