from time import sleep

import gym
import torch
import numpy as np
import slimevolleygym
from torch.distributions import Categorical

# PATH_P = "./model/2024-05-03-14:04:57/model_P.pth"
action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

PATH_P = "./model/2024-07-22-17:49:03/model_A39.pth"
# policy = MultiActionActor(27, 8)
# policy.load_state_dict(torch.load(PATH_P))
policy = torch.load(PATH_P)
# env = gym.make('SlimeVolley-v0', render_mode='human')
env = slimevolleygym.SlimeVolleyEnv()
env.metadata['render_fps'] = 200

tot = 0
for _ in range(100):
    state_i = env.reset()
    # state_i_ = state_i_[0]
    state_j = state_i
    score = 0
    for i in range(1000):
        env.render(mode='human')

        out_i = policy(torch.from_numpy(np.array(state_i)).float().unsqueeze(dim=0))
        out_j = policy(torch.from_numpy(np.array(state_j)).float().unsqueeze(dim=0))

        dist_i = Categorical(out_i)
        dist_j = Categorical(out_j)

        action_i = dist_i.sample()
        action_j = dist_j.sample()

        log_prob_i = dist_i.log_prob(action_i).sum(-1, keepdim=True)
        log_prob_j = dist_j.log_prob(action_j).sum(-1, keepdim=True)

        action_i = action_i.detach().numpy().tolist()
        action_i = action_i[0]
        action_j = action_j.detach().numpy().tolist()
        action_j = action_j[0]
        action_i_ = action_table[action_i]
        action_j_ = action_table[action_j]

        next_state_i_, reward_i, done, info = env.step(action_i_)
        sleep(0.01)
        next_state_j_ = info["otherObs"]

        if done:
            break
    # print(score)
    tot += score
print(tot / 100)
