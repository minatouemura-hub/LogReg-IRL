# IRL with Expert Policy

このプロジェクトは、**CartPole-v1** などの強化学習環境において、

1. **Stable-Baselines3** を使って専門家エージェントをローカルで学習
2. **逆強化学習（IRL）** モデル (`TrainIrlGrid`) を学習
3. 専門家 vs ランダムの累計報酬を比較して、IRL モデルの性能を評価

を一括で実行できるスクリプト群を提供します。

---

## 📦 セットアップ

1. **クローン**

   ```bash
   git clone https://github.com/yourusername/irl-project.git
   cd irl-project
   ```

2. **環境構築**（推奨: `conda`）

   ```bash
   conda create -n irl-env python=3.11
   conda activate irl-env
   ```

3. **依存パッケージのインストール**

   ```bash
   pip install -r requirements.txt
   ```

   * `gymnasium`
   * `stable-baselines3`
   * `torch`
   * `tqdm`
   * その他: `numpy`, `pandas`, `matplotlib`

---

## 🚀 使い方

### 1. 単発実行: 専門家学習 → IRL 推論

```bash
python main.py \
  --env_name CartPole-v1 \
  --algo DQN \
  --train_timesteps 100000 \
  --num_epochs 200 \
  --num_eval 500 \
  --max_steps 200
```

* `--expert_model_path` を**指定しない**場合は、内部で DQN/PPO モデルを学習し `expert_{algo}_{env}.zip` として保存。続けて IRL 推論を実行します。

### 2. 事前学習済みモデルを使う

```bash
python main.py \
  --env_name CartPole-v1 \
  --algo PPO \
  --expert_model_path /path/to/ppo_cartpole.zip \
  --num_epochs 100 \
  --num_eval 300 \
  --max_steps 200
```

* `--expert_model_path` に ZIP ファイルを指定すると、ローカルモデルをロードして IRL 推論のみを実行します。

---

## ⚙️ コマンドラインオプション

| オプション                 | デフォルト         | 説明                                |
| --------------------- | ------------- | --------------------------------- |
| `--env_name`          | `CartPole-v1` | Gym 環境名                           |
| `--algo`              | `DQN`         | 専門家エージェントのアルゴリズム (`DQN` or `PPO`) |
| `--train_timesteps`   | `100000`      | 専門家モデル学習のステップ数                    |
| `--expert_model_path` | `None`        | 事前学習済み ZIP モデルのパス (指定時は学習をスキップ)   |
| `--num_epochs`        | `200`         | IRL モデル学習のエポック数                   |
| `--num_eval`          | `500`         | IRL 評価時のエピソード数                    |
| `--max_steps`         | `200`         | 1エピソードあたりの最大ステップ数                 |

---

## 📂 ディレクトリ構成

```
irl-project/
├── main.py                # エントリーポイント
├── trainer.py             # IRL 学習クラスとデータセット定義
├── validation/valid/      # IRL 推論 & 評価スクリプト
│   └── valid.py
├── logreg/                # 逆強化学習モデルと損失定義
│   ├── models.py
│   └── modules.py
├── requirements.txt       # Python 依存関係
└── README.md              # プロジェクト概要 & 実行手順
```

---

## 🎓 解説

* `TrainIrlGrid` : CartPole-v1 のシミュレーションから専門家／ランダム軌跡を収集し、IRL モデルを学習します。
* `validate_irl_with_open_model`: Stable-Baselines3 で学習済み専門家モデルをロードし、IRL モデルの報酬推定性能を評価します。

---

