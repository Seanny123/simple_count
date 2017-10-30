from vocab import gen_vocab, state_vocab
from mem_net import MemNet
from assoc_net import Cleanup, HeteroMap

import nengo_spa as spa
import nengo
from nengo.presets import ThresholdingEnsembles
import numpy as np

thresh_conf = ThresholdingEnsembles(0.25)

from constants import *

rng = np.random.RandomState(0)
number_dict = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
               "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9}

# This should be set to 9 for the actual final test
max_sum = 9
max_num = max_sum - 2

number_list, num_vocab = gen_vocab(number_dict, max_num, D, rng)

join_num = "+".join(number_list[0:max_num])


with spa.Network(label="Counter", seed=0) as model:
    model.q1 = spa.State(num_vocab)
    model.q2 = spa.State(num_vocab)

    model.answer = spa.State(num_vocab)

    model.op_state = MemNet(state_vocab, label="op_state")

    input_keys = number_list[:max_num-1]
    output_keys = number_list[1:max_num]

    """Result circuit"""
    ## Incrementing memory
    model.res_assoc = HeteroMap(num_vocab, input_keys=input_keys, output_keys=output_keys)
    ## Starting memory
    model.count_res = MemNet(num_vocab, label="count_res")
    ## Increment result memory
    model.res_mem = MemNet(num_vocab, label="res_mem")
    ## Cleanup memory
    model.rmem_assoc = Cleanup(num_vocab)

    """Total circuit"""
    ## Total memory
    model.tot_assoc = HeteroMap(num_vocab, input_keys=input_keys, output_keys=output_keys)
    ## Starting memory
    model.count_tot = MemNet(num_vocab, label="count_tot")
    ## Increment result memory
    model.tot_mem = MemNet(num_vocab, label="tot_mem")
    ## Cleanup memory
    model.tmem_assoc = Cleanup(num_vocab)
    model.ans_assoc = Cleanup(num_vocab)

    ## The memory that says when to stop incrementing
    model.count_fin = MemNet(num_vocab, label="count_fin")

    """Comparison circuit"""
    ## State for easier insertion into Actions after threshold
    model.tot_fin_simi = spa.Scalar()
    model.comp_tot_fin = spa.Compare(D)
    # this network is only used during the on_input action, is it really necessary?
    model.fin_assoc = Cleanup(num_vocab)

    """Compares that set the speed of the increment"""
    ## Compare for loading into start memory
    model.comp_load_res = spa.Compare(D)
    ## Compare for loading into incrementing memory
    model.comp_inc_res = spa.Compare(D)
    ## Cleanup for compare
    model.comp_assoc = Cleanup(num_vocab)

    ## Increment for compare and input
    model.gen_inc_assoc = HeteroMap(num_vocab, input_keys=input_keys, output_keys=output_keys)

    model.running = spa.Transcode(input_vocab=num_vocab)

    # Rule documentation:
    # If the input isn't blank, read it in
    # If not done, prepare next increment
    # If we're done incrementing write it to the answer
    # Increment memory transfer

    main_actions = spa.Actions(f"""
        ifmax (dot(model.q1, {join_num}) + dot(model.q2, {join_num}))*0.5 as 'on_input':
            model.gen_inc_assoc -> model.count_res
            model.q1 -> model.gen_inc_assoc

            ONE -> model.count_tot
            model.fin_assoc -> model.count_fin
            2.5*model.q2 -> model.fin_assoc
            RUN -> model.op_state

        ifmax (model.running - model.tot_fin_simi + 1.25*model.comp_inc_res - model.comp_load_res) as 'cmp_fail':
            0.5*RUN - NONE -> model.op_state

            2.5*model.count_res -> model.rmem_assoc
            2.5*model.count_tot -> model.tmem_assoc

            0 -> model.count_res.gate
            0 -> model.count_tot.gate
            0 -> model.op_state.gate
            0 -> model.count_fin.gate

            model.res_mem -> model.comp_load_res.input_a
            model.comp_assoc -> model.comp_load_res.input_b
            2.5*model.count_res -> model.comp_assoc

        ifmax (0.5*model.running + model.tot_fin_simi) as 'cmp_good':
            8*model.count_res -> model.ans_assoc
            0.5*RUN -> model.op_state

            0 -> model.count_res.gate
            0 -> model.count_tot.gate
            0 -> model.op_state.gate
            0 -> model.count_fin.gate

        ifmax (0.3*model.running + 1.2*model.comp_load_res - model.comp_inc_res) as 'increment':
            2.5*model.res_mem -> model.res_assoc
            2.5*model.tot_mem -> model.tot_assoc

            0 -> model.res_mem.gate
            0 -> model.tot_mem.gate
            0 -> model.op_state.gate
            0 -> model.count_fin.gate

            0.75*ONE -> model.comp_load_res.input_a
            0.75*ONE -> model.comp_load_res.input_b
            model.gen_inc_assoc -> model.comp_inc_res.input_a
            2.5*model.res_mem -> model.gen_inc_assoc
            model.count_res -> model.comp_inc_res.input_b

        always:
            dot(model.op_state, RUN) -> model.running

            model.rmem_assoc -> model.res_mem
            model.tmem_assoc -> model.tot_mem

            model.res_assoc -> model.count_res
            model.tot_assoc -> model.count_tot

            model.count_fin -> model.comp_tot_fin.input_a
            0.5*model.count_tot -> model.comp_tot_fin.input_b
    """)

    """Threshold preventing premature influence from comp_tot_fin similarity"""
    with thresh_conf:
        thresh_ens = nengo.Ensemble(100, 1)

    nengo.Connection(model.comp_tot_fin.output, thresh_ens)
    nengo.Connection(thresh_ens, model.tot_fin_simi.input)

    """Because the answer is being continuously output, we've got to threshold it by the comp_tot_fin similarity"""
    ans_boost = nengo.networks.Product(200, dimensions=D, input_magnitude=2)
    ans_boost.label = "ans_boost"
    nengo.Connection(model.ans_assoc.output, ans_boost.A)
    nengo.Connection(thresh_ens, ans_boost.B,
                     transform=np.ones((D, 1)))
    nengo.Connection(ans_boost.output, model.answer.input, transform=2.5)

# nengo.Connection(env.q_in[D:], model.q1.input)
# nengo.Connection(env.q_in[:D], model.q2.input)
# nengo.Connection(env.op_in, model.op_state.mem.input)
