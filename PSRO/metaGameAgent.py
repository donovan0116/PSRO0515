import numpy as np


def meta_game(args, actor_pop, critic_pop, sample_proportion):
    if args.sample_proportion_mode == 1:
        sample_proportion = np.insert(sample_proportion, 0, 0)
    elif args.sample_proportion_mode == 2:
        # 均匀分布
        n = len(actor_pop)
        sample_proportion = np.full(n, 1/n)
    else:
        # 纳什均衡
        pass
    return sample_proportion
