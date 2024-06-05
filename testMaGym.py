import gym

n_agents = 2

env = gym.make('ma_gym:PongDuel-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
    print(reward_n)
env.close()
print(reward_n)
