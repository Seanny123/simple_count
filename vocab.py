from constants import less_D

import nengo_spa as spa

state_vocab = spa.Vocabulary(less_D)
state_vocab.populate("RUN;NONE")


def gen_vocab(n_dict, n_range=9, dims=32, rng=None):

    vo = spa.Vocabulary(dims, rng=rng)
    n_list = list(n_dict.keys())
    vo.populate(";".join(n_list[:n_range]))

    return n_list, vo
