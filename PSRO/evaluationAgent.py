import os
from multiprocessing import Process

import numpy as np
import redis
import torch
from torch.distributions import Categorical
import pickle
import random
import time

import slimevolleygym
from PSRO.ppo import PPO


class EvaluationAgent:
    def __init__(self, args, actor_training, actor_pop, critic_training, critic_pop, sample_proportion, agent_args,
                 device):
        self.actor_training = actor_training
        self.actor_pop = actor_pop
        self.sample_proportion = sample_proportion

        self.traj_length = agent_args.traj_length
        self.env = args.env
        self.eval_count = args.eval_count
        self.device = device
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.agent_args = agent_args
        self.critic_training = critic_training
        self.critic_pop = critic_pop

        self.action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

        self.num_workers = 10
        self.dis_eval = args.distribute_eval

    def get_action(self, x):
        out = self.actor_training(x)
        return out

    def get_action_pop(self, x, sample_num):
        out = self.actor_pop[sample_num](x)
        return out

    def dis_evaluation(self, state_rms_i, state_rms_j, eval_count):
        winning_rate = 0.0
        r = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=2)
        for key in r.keys():
            r.delete(key)

        redis_conn_2 = redis.ConnectionPool(host='127.0.0.1', port=6379, password='123456', db=2)

        processes = []
        eval_count = eval_count // self.num_workers
        for i in range(self.num_workers):
            p = Process(target=self.one_evaluation, args=(state_rms_i, state_rms_j, eval_count, redis_conn_2))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        redis_client_2 = redis.StrictRedis(connection_pool=redis_conn_2)
        # 从redis中收集所有胜率
        for key in redis_client_2.keys():
            winning_rate_json = redis_client_2.get(key)
            if winning_rate_json:
                winning_rate += pickle.loads(winning_rate_json)

        return winning_rate / self.num_workers

    def one_evaluation(self, state_rms_i, state_rms_j, eval_count, redis_conn_2):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
        win_count = 0
        env_ = slimevolleygym.SlimeVolleyEnv()
        torch.manual_seed(os.getpid() + int(time.time() % 1000000))
        random.seed(os.getpid() + int(time.time() % 1000000))

        for _ in range(eval_count):

            sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()

            state_i_ = env_.reset()
            # state_i_ = state_i_[0]
            state_j_ = state_i_
            state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
            state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
            tot_reward_i, tot_reward_j = 0, 0
            for step in range(self.traj_length):
                out_i = self.get_action(torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0))
                out_j = self.get_action_pop(torch.from_numpy(np.array(state_j)).float().to(self.device).unsqueeze(dim=0)
                                            , sample_num)

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
                action_i_ = self.action_table[action_i]
                action_j_ = self.action_table[action_j]
                # next_state_i_, reward_i, done, info = self.env.step(action_i_)
                next_state_i_, reward_i, done, info = env_.step(action_i_, action_j_)
                tot_reward_i += reward_i
                next_state_j_ = info["otherObs"]
                next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                next_state_j = np.clip((next_state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                if reward_i == 1.01:
                    reward_j = -0.99
                elif reward_i == -0.99:
                    reward_j = 1.01
                else:
                    reward_j = 0.01
                tot_reward_j += reward_j
                if done or step == self.traj_length - 1:
                    # 如果i胜利，则存入win_count + 1
                    if tot_reward_i >= tot_reward_j:
                        win_count += 1
                    break
                else:
                    state_i = next_state_i
                    state_j = next_state_j
                    state_i_ = next_state_i_
                    state_j_ = next_state_j_
        if self.dis_eval:
            # 将胜率上传
            redis_client_2 = redis.StrictRedis(connection_pool=redis_conn_2)
            key = f"{os.getpid()}"
            winning_rate_json = pickle.dumps(win_count / eval_count)
            redis_client_2.set(key, winning_rate_json)

        return win_count / self.eval_count

    def evaluation(self, state_rms_i, state_rms_j):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
        win_count = 0
        # env_ = slimevolleygym.SlimeVolleyEnv()
        # torch.manual_seed(os.getpid() + int(time.time() % 1000000))
        # random.seed(os.getpid() + int(time.time() % 1000000))

        for _ in range(self.eval_count):

            sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()

            state_i_ = self.env.reset()
            # state_i_ = state_i_[0]
            state_j_ = state_i_
            state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
            state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
            tot_reward_i, tot_reward_j = 0, 0
            for step in range(self.traj_length):
                out_i = self.get_action(torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0))
                out_j = self.get_action_pop(torch.from_numpy(np.array(state_j)).float().to(self.device).unsqueeze(dim=0)
                                            , sample_num)

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
                action_i_ = self.action_table[action_i]
                action_j_ = self.action_table[action_j]
                # next_state_i_, reward_i, done, info = self.env.step(action_i_)
                next_state_i_, reward_i, done, info = self.env.step(action_i_, action_j_)
                tot_reward_i += reward_i
                next_state_j_ = info["otherObs"]
                next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                next_state_j = np.clip((next_state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                if reward_i == 1.01:
                    reward_j = -0.99
                elif reward_i == -0.99:
                    reward_j = 1.01
                else:
                    reward_j = 0.01
                tot_reward_j += reward_j
                if done or step == self.traj_length - 1:
                    # 如果i胜利，则存入win_count + 1
                    if tot_reward_i >= tot_reward_j:
                        win_count += 1
                    break
                else:
                    state_i = next_state_i
                    state_j = next_state_j
                    state_i_ = next_state_i_
                    state_j_ = next_state_j_
        # if self.dis_eval:
        #     # 将胜率上传
        #     redis_client_2 = redis.StrictRedis(connection_pool=redis_conn_2)
        #     key = f"{os.getpid()}"
        #     winning_rate_json = pickle.dumps(win_count / eval_count)
        #     redis_client_2.set(key, winning_rate_json)

        return win_count / self.eval_count

    def evaluation_sota(self, state_rms_i, state_rms_j, sample_proportion, actor_pop):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(sample_proportion).to(self.device).float().detach()
        win_count = 0
        # self.actor_pop = actor_pop

        for _ in range(self.eval_count):
            sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()
            state_i_ = self.env.reset()
            # state_i_ = state_i_[0]
            state_i = state_i_
            # state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
            tot_reward_i, tot_reward_j = 0, 0
            for step in range(self.traj_length):
                # out_i = self.get_action_pop(torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0)
                #                             , sample_num)
                out_i = actor_pop[sample_num](
                    torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0))

                dist_i = Categorical(out_i)

                action_i = dist_i.sample()

                action_i = action_i.detach().numpy().tolist()
                action_i = action_i[0]

                action_i_ = self.action_table[action_i]

                next_state_i_, reward_i, done, info = self.env.step(action_i_)
                tot_reward_i += reward_i
                # next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                next_state_i = next_state_i_
                if reward_i == 1.01:
                    reward_j = -0.99
                elif reward_i == -0.99:
                    reward_j = 1.01
                else:
                    reward_j = 0.01
                tot_reward_j += reward_j
                if done or step == self.traj_length - 1:
                    # 如果i胜利，则存入win_count + 1
                    if tot_reward_i >= tot_reward_j:
                        win_count += 1
                    break
                else:
                    state_i = next_state_i

        return win_count / self.eval_count
