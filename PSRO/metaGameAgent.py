import numpy as np
import scipy

from PSRO.evaluationAgent import EvaluationAgent


def computing_nash_equilibrium(len_actor_pop, winning_rate_table, *args, **kwargs):
    payoff_matrix = np.ones((len_actor_pop, len_actor_pop), dtype=float)
    """
    反对称化 A + AT = 0
    """
    for i in range(0, len_actor_pop):
        payoff_matrix[i][i] = 0.
        for j in range(0, i):
            payoff_matrix[i][j] = np.log((winning_rate_table[i][j] + 1e-10) / (1 - winning_rate_table[i][j] + 1e-10))
            payoff_matrix[j][i] = -payoff_matrix[i][j]
    row_count, col_count = payoff_matrix.shape

    # Variables: Row strategy weights, value of the game.

    # Objective: Maximize the minimum possible row player's payoff.
    c = np.zeros((row_count + 1))
    c[-1] = -1.0  # SciPy uses the minimization convention.

    # Payoff when column player plays any strategy must be at least the value of the game.
    A_ub = np.concatenate((-payoff_matrix.transpose(), np.ones((col_count, 1))), axis=1)
    b_ub = np.zeros(col_count)

    # Probabilities must sum to 1.
    A_eq = np.ones((1, row_count + 1))
    A_eq[0, -1] = 0

    b_eq = np.ones((1))

    # Weights must be nonnegative. Payoff must be between the minimum and maximum value in the payoff matrix.
    min_payoff = np.min(payoff_matrix)
    max_payoff = np.max(payoff_matrix)
    bounds = [(0, None)] * row_count + [(min_payoff, max_payoff)]

    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, *args, **kwargs)

    result.strategy = result.x[:-1]
    result.value = result.x[-1]
    return result.strategy  # ndarray, vector


def meta_game(args, actor_pop, critic_pop, sample_proportion, agent_args, device, state_rms_i, state_rms_j,
              winning_rate_table):
    new_winning_rate_table = winning_rate_table
    if args.sample_proportion_mode == 1:
        sample_proportion = np.insert(sample_proportion, 0, 0)
    elif args.sample_proportion_mode == 2:
        # 均匀分布
        n = len(actor_pop)
        sample_proportion = np.full(n, 1 / n)
    else:
        # 纳什均衡
        # 返回sample pro
        # 输入actor_pop和critic_pop，其中最后一个是新加入的，要使用最后一个和其他所有的比100次，保存胜率
        winning_rate_list = []
        for idx in range(len(actor_pop) - 1):
            evaluationAgent = EvaluationAgent(args, actor_pop[-1], [actor_pop[idx]], critic_pop[-1],
                                              [critic_pop[idx]], np.array([1.]), agent_args, device)
            winning_rate = evaluationAgent.evaluation(state_rms_i, state_rms_j)
            winning_rate_list.append(winning_rate)

        n = winning_rate_table.shape[0]
        new_winning_rate_table = np.zeros((n + 1, n + 1))
        new_winning_rate_table[:n, :n] = winning_rate_table

        new_winning_rate_table[n, :n] = winning_rate_list  # 添加到最后一行
        new_winning_rate_table[:n, n] = winning_rate_list  # 添加到最后一列
        # 计算纳什均衡
        sample_proportion = computing_nash_equilibrium(len(actor_pop), new_winning_rate_table)
    return sample_proportion, new_winning_rate_table
