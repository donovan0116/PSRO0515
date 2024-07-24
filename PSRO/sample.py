from copy import deepcopy

import numpy as np
import torch
from torch.distributions import Categorical

import slimevolleygym
from Utils.utils import make_transition, RunningMeanStd, count_frequencies, ReplayBuffer, timing_function

import multiprocessing

import pickle
import redis
import time
from multiprocessing import Process, Pipe, connection
import os
import random

'''
思路：构建sample的agent，采样完成后将data传出去，再在train中创建agent进行训练。
'''


class SampleAgent:
    def __init__(self, args, actor_pop, critic_pop, actor_training, critic_training, sample_proportion, agent_args,
                 device):
        self.traj_length = agent_args.traj_length
        self.batch_size = agent_args.batch_size
        self.mini_batch_num = args.mini_batch_num
        self.device = device

        self.agent_args = agent_args
        self.data = ReplayBuffer(action_prob_exist=True, max_size=self.traj_length, state_dim=args.state_dim,
                                 num_action=1)

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim

        self.env = args.env
        self.actor_pop = actor_pop
        self.critic_pop = critic_pop
        self.actor_training = actor_training
        self.critic_training = critic_training
        self.sample_proportion = sample_proportion
        self.reward_scaling = args.reward_scaling

        self.action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

        # self.num_workers = multiprocessing.cpu_count()
        self.num_workers = 8

    def get_action(self, x):
        out = self.actor_training(x)
        return out

    def get_action_pop(self, x, sample_num):
        out = self.actor_pop[sample_num](x)
        return out

    def rollout(self, t, env_, num_rollout, state_i, state_j, sample_num, state_rms_i, state_rms_j):
        out_i = self.get_action(torch.from_numpy(np.array(state_i)).float().to(self.device).unsqueeze(dim=0))
        out_j = self.get_action_pop(torch.from_numpy(np.array(state_j)).float().to(self.device).unsqueeze(dim=0),
                                    sample_num)

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
        next_state_j_ = info["otherObs"]
        dw = done
        if t == num_rollout - 1 and done == 0:
            dw = 1 - dw
        next_state_i = np.clip((next_state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
        next_state_j = np.clip((next_state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
        transition_i = make_transition(state_i,
                                       action_i,
                                       np.array([reward_i * self.reward_scaling]),
                                       next_state_i,
                                       np.array([done]),
                                       np.array([dw]),
                                       log_prob_i.detach().cpu().numpy()
                                       )

        return done, next_state_i, next_state_j, next_state_i_, next_state_j_, transition_i

    # @timing_function
    def sample(self, state_i_lst, state_rms_i, state_j_lst, state_rms_j):
        # 通过sample_proportion选择actor_pop
        sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
        sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()

        num_rollout = self.traj_length
        state_i_ = self.env.reset()
        # state_i_ = state_i_[0]
        state_j_ = state_i_
        state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
        state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
        temp = 0
        for t in range(num_rollout):
            state_i_lst.append(state_i_)
            state_j_lst.append(state_j_)

            done, next_state_i, next_state_j, next_state_i_, next_state_j_, transition_i \
                = self.rollout(t, self.env, num_rollout, state_i, state_j, sample_num, state_rms_i, state_rms_j)
            self.data.put_data(transition_i)

            if done:
                state_i_ = self.env.reset()
                # state_i_ = state_i_[0]
                state_j_ = state_i_
                state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                print(t - temp)
                temp = t
            else:
                state_i = next_state_i
                state_j = next_state_j
                state_i_ = next_state_i_
                state_j_ = next_state_j_

        # result = count_frequencies(action_lst)
        # print(f"the frequencies of action is: {result}")
        # print(f"the frequencies of action is: {out_i}")

        result = count_frequencies(self.data.data['action'])
        print(f"the frequencies of action is: {result}")
        return self.data

    # 分布式采样

    # 线程执行函数
    def worker_process(self, remote: connection.Connection, seed: int, redis_conn_0, redis_conn_1, state_rms_i,
                       state_rms_j):
        """
        Worker process to interact with the environment and store data in Redis.
        """
        redis_client_0 = redis.StrictRedis(connection_pool=redis_conn_0)
        redis_client_1 = redis.StrictRedis(connection_pool=redis_conn_1)
        torch.manual_seed(os.getpid() + int(time.time() % 1000000))
        random.seed(os.getpid() + int(time.time() % 1000000))
        env_ = self.env

        while True:
            cmd, data = remote.recv()
            if cmd == "run_episode":
                # print(f"{os.getpid()}" + "is sampling...")
                # num_rollout = self.traj_length // self.num_workers
                num_rollout = self.traj_length
                # 通过sample_proportion选择actor_pop
                sample_pro = torch.from_numpy(self.sample_proportion).to(self.device).float().detach()
                sample_num = Categorical(sample_pro).sample().detach().numpy().tolist()
                # state 正则化
                state_i_ = env_.reset()
                state_j_ = state_i_
                state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                state_i_lst, state_j_lst = [], []
                for t in range(num_rollout):
                    state_i_lst.append(state_i_)
                    state_j_lst.append(state_j_)

                    done, next_state_i, next_state_j, next_state_i_, next_state_j_, transition_i \
                        = self.rollout(t, env_, num_rollout, state_i, state_j, sample_num, state_rms_i,
                                       state_rms_j)
                    # 将生成好的traj存入redis中
                    key = f"{seed}_{t}"
                    traj_json = pickle.dumps(transition_i)
                    redis_client_0.set(key, traj_json)

                    if done or t == num_rollout - 1:
                        if len(redis_client_0.keys()) >= self.traj_length:
                            break
                        state_i_ = env_.reset()
                        state_j_ = state_i_
                        state_i = np.clip((state_i_ - state_rms_i.mean) / (state_rms_i.var ** 0.5 + 1e-8), -5, 5)
                        state_j = np.clip((state_j_ - state_rms_j.mean) / (state_rms_j.var ** 0.5 + 1e-8), -5, 5)
                    else:
                        state_i = next_state_i
                        state_j = next_state_j
                        state_i_ = next_state_i_
                        state_j_ = next_state_j_
                # print(f"{os.getpid()}" + "is over...")
                # 将更新的lst存到redis中，因为每个worker事实上只用了一次所以没事不会覆盖
                key_lst_i = f"{seed}_lst_i"
                lst_json_i = pickle.dumps(state_i_lst)
                redis_client_1.set(key_lst_i, lst_json_i)

                key_lst_j = f"{seed}_lst_j"
                lst_json_j = pickle.dumps(state_j_lst)
                redis_client_1.set(key_lst_j, lst_json_j)
                remote.send("done")
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError

    # @timing_function
    def dis_sample(self, state_i_lst, state_rms_i, state_j_lst, state_rms_j):
        r = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=0)
        for key in r.keys():
            r.delete(key)
        r = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=1)
        for key in r.keys():
            r.delete(key)

        redis_conn_0 = redis.ConnectionPool(host='127.0.0.1', port=6379, password='123456', db=0)
        redis_conn_1 = redis.ConnectionPool(host='127.0.0.1', port=6379, password='123456', db=1)
        # 线程池
        workers = []
        for i in range(self.num_workers):
            worker = Worker(i, redis_conn_0, redis_conn_1, self.worker_process, state_rms_i, state_rms_j)
            workers.append(worker)

        redis_client_0 = redis.StrictRedis(connection_pool=redis_conn_0)
        redis_client_1 = redis.StrictRedis(connection_pool=redis_conn_1)

        processes = []
        for worker in workers:
            p = Process(target=worker.run_episode)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 关闭所有worker
        for worker in workers:
            worker.close()

        def sort_key(s):
            s = s.decode('utf-8')
            part1, part2 = s.split('_')
            return (int(part1), int(part2))

        # 拉取所有key中的traj
        keys = redis_client_0.keys()

        # 使用sorted函数和自定义key进行排序
        keys = sorted(keys, key=sort_key)
        for k in range(self.traj_length):
            traj_json = redis_client_0.get(keys[k])
            if traj_json:
                traj = pickle.loads(traj_json)
                self.data.put_data(traj)

        # for w in range(self.num_workers):
        #     for t in range(self.traj_length // self.num_workers):
        #         key = f"{w}_{t}"
        #         traj_json = redis_client_0.get(key)
        #         if traj_json:
        #             traj = pickle.loads(traj_json)
        #             self.data.put_data(traj)

        # 拉取所有state_lst
        for w in range(self.num_workers):
            key_i = f"{w}_lst_i"
            key_j = f"{w}_lst_j"
            state_i_lst = state_i_lst + pickle.loads(redis_client_1.get(key_i))
            state_j_lst = state_j_lst + pickle.loads(redis_client_1.get(key_j))

        result = count_frequencies(self.data.data['action'])
        print(f"the frequencies of action is: {result}")
        return self.data, state_i_lst, state_j_lst


class Worker:
    def __init__(self, seed, redis_conn_0, redis_conn_1, worker_process, state_rms_i, state_rms_j):
        self.child, parent = Pipe()
        self.process = Process(target=worker_process,
                               args=(parent, seed, redis_conn_0, redis_conn_1, state_rms_i, state_rms_j))
        self.process.start()

    def run_episode(self):
        self.child.send(("run_episode", None))
        return self.child.recv()

    def close(self):
        self.child.send(("close", None))
        self.process.join()
