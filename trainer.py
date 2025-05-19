import os

import gymnasium as gym
import torch
from torch.nn import Module
from torch.optim import Adagrad
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def default_policy(state):
    """CartPole-v1 用のシンプルルールベース専門家ポリシー"""
    if isinstance(state, tuple):
        state = state[0]
    theta = state[2]
    return 1 if theta > 0 else 0


class GridIRLDataset(Dataset):
    def __init__(self, transitions):
        # transitions: list of (state, next_state, label)
        self.data = transitions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, s_next, label = self.data[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class TrainIrlGrid(Module):
    def __init__(
        self,
        env_name,
        expert_policy,
        num_epoch=100,
        gamma=0.95,
        lr=0.01,
        lambda_reg=0.05,
        state_space=None,
    ):
        super().__init__()
        self.env = gym.make(env_name)
        self.expert_policy = expert_policy
        self.num_epoch = num_epoch
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ユーザ指定 or データから推定
        self.state_space = state_space

        self.irl_model = None
        self.optimizers = []

    def collect_transitions(self, num_expert=1000, num_random=1000, max_steps=50):
        transitions = []
        # 専門家軌跡
        for _ in range(num_expert):
            state, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.expert_policy(state)
                step = self.env.step(action)
                if len(step) == 5:
                    next_state, _, terminated, truncated, _ = step
                    done = terminated or truncated
                else:
                    next_state, _, done, _ = step
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                transitions.append((state, next_state, 1))
                state = next_state
                if done:
                    break

        # ランダム軌跡
        for _ in range(num_random):
            state, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                step = self.env.step(action)
                if len(step) == 5:
                    next_state, _, terminated, truncated, _ = step
                    done = terminated or truncated
                else:
                    next_state, _, done, _ = step
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                transitions.append((state, next_state, -1))
                state = next_state
                if done:
                    break

        return transitions

    def build_model(self, state_dim, hidden_dim=64):
        from logreg.models import Irl_Net

        self.irl_model = Irl_Net(
            device=self.device,
            action_num=self.env.action_space.n,
            state_space=state_dim,
        ).to(self.device)

        # Optimizers for each head
        self.optimizers = [
            Adagrad(self.irl_model.dens_net.parameters(), lr=self.lr),
            Adagrad(self.irl_model.reward_net.parameters(), lr=self.lr),
            Adagrad(self.irl_model.state_value_net.parameters(), lr=self.lr),
        ]

    def train(self):
        # 1) Collect data
        transitions = self.collect_transitions()
        dataset = GridIRLDataset(transitions)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 2) Determine state_space if not set
        if self.state_space is None:
            sample_state, _, _ = dataset[0]
            self.state_space = sample_state.numel()
        state_dim = self.state_space

        # 3) Build model
        self.build_model(state_dim)

        # 4) Training loop
        best_loss = float("inf")
        for epoch in tqdm(range(self.num_epoch), desc="Epochs"):
            total_loss = 0.0
            self.irl_model.train()

            for s, s_next, label in dataloader:
                s = s.to(self.device)
                s_next = s_next.to(self.device)
                label = label.unsqueeze(1).to(self.device)

                # Pseudo-image reshape (B, state_dim) -> (B, state_dim, 1, 1)
                s = s.unsqueeze(-1).unsqueeze(-1)
                s_next = s_next.unsqueeze(-1).unsqueeze(-1)

                dens, q_hat, v_x, v_y = self.irl_model(s, s_next)
                from logreg.modules import negative_log_likelihood

                # Correct argument order: outputs, labels, model, lambda_reg
                dens_loss = negative_log_likelihood(
                    dens,
                    label,
                    self.irl_model.dens_net,
                    self.lambda_reg,
                )
                log_ratio = torch.sigmoid(dens * label) + q_hat + self.gamma * v_y - v_x
                q_loss = negative_log_likelihood(
                    log_ratio,
                    label,
                    self.irl_model.reward_net,
                    self.lambda_reg,
                )
                v_loss = negative_log_likelihood(
                    log_ratio,
                    label,
                    self.irl_model.state_value_net,
                    self.lambda_reg,
                )
                loss = dens_loss + q_loss + v_loss

                for opt in self.optimizers:
                    opt.zero_grad()
                loss.backward()
                for opt in self.optimizers:
                    opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            tqdm.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss

        return self.irl_model


# Alias for external import
expert_policy = default_policy
