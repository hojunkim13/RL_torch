import numpy as np
import torch
from torch.optim import Adam
from network import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, env_name, state_dim, action_dim, lr, gamma):
        self.env = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)

    def getAction(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = state.unsqueeze(0)
        with torch.no_grad():
            probs = self.net(state)[0].squeeze().cpu().numpy()
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def train(self, s, a, r, s_, d):
        self.optimizer.zero_grad()
        s = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)
        a = torch.eye(self.action_dim)[a].to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device).unsqueeze(0)
        s_ = torch.tensor(s_, dtype=torch.float32).to(device).unsqueeze(0)
        d = torch.tensor(d, dtype=torch.bool).to(device).unsqueeze(0)

        probs, value = self.net(s)
        value_ = self.net(s_)[1]

        critic_target = r + value_ * self.gamma * ~d
        critic_loss = torch.mean(torch.square(value - critic_target))

        action_prob = torch.sum(probs * a).clip(1e-8, None)
        advantage = (
            critic_target - value
        )  # difference between pred value and real value
        actor_loss = -torch.log(action_prob) * advantage

        loss = critic_loss + actor_loss
        loss.backward()
        self.optimizer.step()

    def save(self, PATH, e):
        path = PATH + self.env + f"_{e}.pth"
        torch.save(self.net.state_dict(), path)

    def load(self, PATH, e):
        path = PATH + self.env + f"_{e}.pth"
        self.net.load_state_dict(torch.load(path))
