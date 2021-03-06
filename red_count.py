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

    # Starting point
    model.q1 = spa.Transcode(output_vocab=num_vocab)
    # The memory that says when to stop incrementing
    model.count_fin = spa.Transcode(output_vocab=num_vocab)

    model.answer = spa.Transcode(output_vocab=num_vocab)

    model.op_state = MemNet(state_vocab, label="op_state")

    input_keys = number_list[:max_num-1]
    output_keys = number_list[1:max_num]

    with spa.Network(label="Result") as res:
        # Incrementing memory
        res.res_assoc = HeteroMap(num_vocab, input_keys=input_keys, output_keys=output_keys)
        # Starting memory
        res.count_res = MemNet(num_vocab, label="count_res")
        # Increment result memory
        res.res_mem = MemNet(num_vocab, label="res_mem")
        # Cleanup memory
        res.rmem_assoc = Cleanup(num_vocab)

    with spa.Network(label="Final Goal") as fin:
        # State for easier insertion into Actions after threshold
        fin.res_fin_simi = spa.Scalar()
        fin.comp_res_fin = spa.Compare(num_vocab)

    with spa.Network(label="Increment Control") as inc_comp:
        """Compares that set the speed of the increment"""
        # Compare for loading into start memory
        inc_comp.comp_load_res = spa.Compare(num_vocab)
        # Compare for loading into incrementing memory
        inc_comp.comp_inc_res = spa.Compare(num_vocab)
        # Cleanup for compare
        inc_comp.comp_assoc = Cleanup(num_vocab)

    # Increment for compare and input
    model.gen_inc_assoc = HeteroMap(num_vocab, input_keys=input_keys, output_keys=output_keys)

    model.running = spa.Scalar()

    # Rule documentation:
    # If the input isn't blank, read it in
    # If not done, prepare next increment
    # If we're done incrementing write it to the answer
    # Increment memory transfer

    main_actions = spa.Actions(f"""
        ifmax dot(model.q1, {join_num})*0.5 as 'on_input':
            model.q1 -> model.gen_inc_assoc
            model.gen_inc_assoc -> res.count_res

            RUN -> model.op_state

        elifmax (model.running - fin.tot_fin_simi + 1.25*inc_comp.comp_inc_res - inc_comp.comp_load_res) as 'cmp_fail':
            0.5*RUN - NONE -> model.op_state

            2.5*res.count_res -> res.rmem_assoc

            0 -> res.count_res.gate
            0 -> model.op_state.gate

            res.res_mem -> inc_comp.comp_load_res.input_a
            inc_comp.comp_assoc -> inc_comp.comp_load_res.input_b
            2.5*res.count_res -> inc_comp.comp_assoc
        
        elifmax (0.5*model.running + fin.tot_fin_simi) as 'cmp_good':
            8*res.count_res -> tot.ans_assoc
            0.5*RUN -> model.op_state

            0 -> res.count_res.gate
            0 -> tot.count_tot.gate
            0 -> model.op_state.gate
            0 -> fin.count_fin.gate

        elifmax (0.3*model.running + 1.2*inc_comp.comp_load_res - inc_comp.comp_inc_res) as 'increment':
            2.5*res.res_mem -> res.res_assoc
            2.5*tot.tot_mem -> tot.tot_assoc

            0 -> res.res_mem.gate
            0 -> tot.tot_mem.gate
            0 -> model.op_state.gate
            0 -> fin.count_fin.gate

            0.75*ONE -> inc_comp.comp_load_res.input_a
            0.75*ONE -> inc_comp.comp_load_res.input_b
            model.gen_inc_assoc -> inc_comp.comp_inc_res.input_a
            2.5*res.res_mem -> model.gen_inc_assoc
            res.count_res -> inc_comp.comp_inc_res.input_b

        always:
            dot(model.op_state, RUN) -> model.running

            res.rmem_assoc -> res.res_mem

            res.res_assoc -> res.count_res

            fin.count_fin -> fin.comp_res_fin.input_a
            0.5*res.count_res -> fin.comp_res_fin.input_b
    """)

    """Threshold preventing premature influence from comp_tot_fin similarity"""
    with thresh_conf:
        thresh_ens = nengo.Ensemble(100, 1)

    nengo.Connection(fin.comp_res_fin.output, thresh_ens)
    nengo.Connection(thresh_ens, fin.res_fin_simi.input)

    """Because the answer is being continuously output, we've got to threshold it by the comp_tot_fin similarity"""
    ans_boost = nengo.networks.Product(200, dimensions=D, input_magnitude=2)
    ans_boost.label = "ans_boost"
    nengo.Connection(res.ans_assoc.output, ans_boost.A)
    nengo.Connection(thresh_ens, ans_boost.B, transform=np.ones((D, 1)))
    nengo.Connection(ans_boost.output, model.answer.input, transform=2.5)

# for the Nengo GUI
for net in model.all_networks:
    if net.label is not None and "->" in net.label:
        net.label = ""

# with nengo.Simulator(model) as sim:
#     sim.run(5)

main_actions[0].bg.label = "bg"
