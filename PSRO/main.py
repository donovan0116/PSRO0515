import argparse
import logging
from configparser import ConfigParser

import copy
import os
from datetime import datetime

import redis
import torch
import gym
import slimevolleygym
import numpy as np
import multiprocessing as mp

from torch.distributions import Categorical

from Networks.network import Actor, Critic
from PSRO.ppo import PPO
from sample import SampleAgent
from train import TrainAgent
from evaluationAgent import EvaluationAgent
from metaGameAgent import meta_game

from Utils.utils import make_transition, Dict, RunningMeanStd
from time import sleep


def are_parameters_updated(model_before, model_after):
    """
    比较训练前后的模型参数是否更新。

    参数:
    model_before (nn.Module): 训练前的模型。
    model_after (nn.Module): 训练后的模型。

    返回:
    bool: 如果参数更新了返回 True，否则返回 False。
    """
    for param_before, param_after in zip(model_before.parameters(), model_after.parameters()):
        if not torch.equal(param_before, param_after):
            return True
    return False


action_table = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")

    parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
    parser.add_argument("--state_dim", type=int, default=12, help="state_dim")
    parser.add_argument("--action_dim", type=int, default=6, help="action_dim")
    parser.add_argument("--env", type=int, default=slimevolleygym.SlimeVolleyEnv(), help="environment")

    parser.add_argument("--max_step_per_episode", type=int, default=3000, help="max_step_per_episode")
    parser.add_argument("--mini_batch_num", type=int, default=10, help="split batch into 10 part")

    parser.add_argument("--eval_count", type=int, default=500, help="evaluation count")
    parser.add_argument("--sample_proportion_mode", type=int, default=3,
                        help="sample_proportion mode 1: SP 2: Uniform Distribution 3: nash")

    parser.add_argument("--lr_meta", type=float, default=5e-3, help="Learning rate of sample proportion")
    parser.add_argument("--battle_episodes_for_winning_rate_matrix", type=int, default=100,
                        help="battle_episodes_for_winning_rate_matrix")
    parser.add_argument("--winning_rate_threshold_for_policy_improvement", type=int, default=0.9,
                        help="winning_rate_threshold_for_policy_improvement")

    parser.add_argument("--max_winning_rate", type=float, default=0.9, help="max_winning_rate")
    parser.add_argument("--max_actor_training_num", type=float, default=3000, help="max_winning_rate")
    parser.add_argument("--reward_scaling", type=float, default=0.1, help='reward scaling(default : 0.1)')

    parser.add_argument("--distribute_sample", type=bool, default=True, help="Trick: distribute sample")
    parser.add_argument("--distribute_eval", type=bool, default=True, help="Trick: distribute evaluation")

    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor_pop = [Actor(agent_args.layer_num,
                       args.state_dim,
                       args.action_dim,
                       agent_args.hidden_dim,
                       agent_args.activation_function,
                       agent_args.last_activation,
                       agent_args.trainable_std
                       )]
    critic_pop = [Critic(agent_args.layer_num,
                         args.state_dim,
                         1,
                         agent_args.hidden_dim,
                         agent_args.activation_function,
                         agent_args.last_activation
                         )]

    sample_proportion = np.array([1.])
    winning_rate_table = np.zeros((1, 1))

    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # 配置日志记录
    logging.basicConfig(level=logging.DEBUG,  # 设置日志级别
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
                        handlers=[logging.FileHandler(f"{current_time}.log"),  # 将日志输出到文件
                                  logging.StreamHandler()])  # 同时将日志输出到控制台

    # 创建日志记录器
    logger = logging.getLogger(__name__)

    test_mode = False

    if test_mode:
        policy = torch.load("./model/2024-06-29-16:40:13/model_A7.pth")
        env = slimevolleygym.SlimeVolleyEnv()

        tot = 0
        for _ in range(100):
            state_i = env.reset()
            sleep(0.02)  # 0.01
            # state_i_ = state_i_[0]
            state_j = state_i
            score = 0
            for i in range(1000):

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

                next_state_i_, reward_i, done, info = env.step(action_i_, action_j_)
                next_state_j_ = info["otherObs"]
                env.render(mode='human')
                sleep(0.05)
                if done:
                    sleep(2)
                    break
                else:
                    state_i = next_state_i_
                    state_j = next_state_j_
            # print(score)
            tot += reward_i
        print(tot / 100)

    while True:
        actor_training = Actor(agent_args.layer_num,
                               args.state_dim,
                               args.action_dim,
                               agent_args.hidden_dim,
                               agent_args.activation_function,
                               agent_args.last_activation,
                               agent_args.trainable_std
                               )
        critic_training = Critic(agent_args.layer_num,
                                 args.state_dim,
                                 1,
                                 agent_args.hidden_dim,
                                 agent_args.activation_function,
                                 agent_args.last_activation
                                 )
        flag = 0
        state_i_lst, state_j_lst = [], []
        state_rms_i = RunningMeanStd(args.state_dim)
        state_rms_j = RunningMeanStd(args.state_dim)

        for n_epi in range(args.max_actor_training_num):
            logger.info('#######################################################')
            logger.info("generation: " + str(len(actor_pop)) + "   training time: " + str(n_epi))
            # sample
            logger.info("sampling...")
            sampleAgent = SampleAgent(args, actor_pop, critic_pop, actor_training, critic_training, sample_proportion,
                                      agent_args, device)
            if args.distribute_sample:
                buffer_i, state_i_lst, state_j_lst = sampleAgent.dis_sample(state_i_lst, state_rms_i, state_j_lst,
                                                                            state_rms_j)
            else:
                buffer_i = sampleAgent.sample(state_i_lst, state_rms_i, state_j_lst, state_rms_j)
            # training
            logger.info("training...")
            agent = PPO(device, args.state_dim, args.action_dim, agent_args, actor_training, critic_training, buffer_i)
            agent.train_net(n_epi)
            # trainAgent = TrainAgent(args, actor_training, critic_training, buffer_i, agent_args, device)
            # trainAgent.train(n_epi)
            state_rms_i.update(np.vstack(state_i_lst))
            state_rms_j.update(np.vstack(state_j_lst))
            evaluationAgent = EvaluationAgent(args, actor_training, actor_pop, critic_training, critic_pop,
                                              sample_proportion, agent_args, device)
            logger.info("evaluating...")
            if args.distribute_eval:
                winning_rate = evaluationAgent.dis_evaluation(state_rms_i, state_rms_j, args.eval_count)
            else:
                winning_rate = evaluationAgent.evaluation(state_rms_i, state_rms_j)
            logger.info(f"winning rate: {winning_rate}")
            logger.info("#######################################################")
            flag = n_epi
            if winning_rate > args.max_winning_rate:
                path = "./model/" + current_time
                if not os.path.exists(path):
                    os.makedirs(path)
                PATH_A = path + "/model_A" + str(len(actor_pop)) + ".pth"
                torch.save(actor_training, PATH_A)
                logger.info(f"generation {str(len(actor_pop))} saved successfully...")
                actor_pop.append(actor_training)
                critic_pop.append(critic_training)
                sample_proportion, winning_rate_table = meta_game(args, actor_pop, critic_pop, sample_proportion,
                                                                  agent_args, device, state_rms_i, state_rms_j,
                                                                  winning_rate_table)

                logger.info("new sample prop: " + str(sample_proportion))
                np.save('./sample_prop.npy', sample_proportion)
                # 每隔5次，与sota对比评估一次
                if len(actor_pop) % 5 == 0:
                    winning_rate_sota = evaluationAgent.evaluation_sota(state_rms_i, state_rms_j, sample_proportion
                                                                        , actor_pop)
                    logger.info("winning rate with sota: " + str(winning_rate_sota))
                break
        if flag == args.max_actor_training_num - 1:
            logger.info("training finished")
            break
