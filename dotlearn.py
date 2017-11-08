import nengo
import nengo_spa as spa
import nengo_gui

import matplotlib.pyplot as plt

import numpy as np

dimensions = 16
t_stop = 2
seedy = 3
total_time = 3

cycle_length = 0.6

vocab = spa.Vocabulary(dimensions, np.random.RandomState(seedy))

vocab.populate("A; B; C")

def objective_func(t, x):
    if t > t_stop:
        return 0
    else:
        return x[0] - np.dot(x[1:], vocab["B"].v)

def cycles(t):
    c = t % cycle_length
    if c < cycle_length / 3.0:
        return "A"
    elif c < 2 * cycle_length / 3.0 :
        return "B"
    else:
        return "C"

with spa.Network(seed=seedy) as model:
    model.input = spa.Transcode(cycles, output_vocab=vocab)

    model.state = spa.State(vocab, subdimensions=dimensions, represent_identity=False)
    assert len(model.state.all_networks) == 1
    assert len(model.state.all_networks[0].ea_ensembles) == 1
    state_ens = model.state.all_networks[0].ea_ensembles[0]

    model.output = nengo.Node(size_in=1, label="dot output")

    model.error = nengo.Node(objective_func, size_in=dimensions+1, size_out=1)

    nengo.Connection(model.input.output, model.state.input, synapse=None)

    nengo.Connection(state_ens, model.error[1:])
    nengo.Connection(model.output, model.error[0])

    # TODO: Connect from the neurons for the learned connection somehow
    # conditions

    conn = nengo.Connection(state_ens, model.output,
                            transform=np.zeros((1, dimensions)),
                            learning_rule_type=nengo.PES())

    nengo.Connection(model.error, conn.learning_rule)

    error_probe = nengo.Probe(model.error)
    output_probe = nengo.Probe(model.output)
    input_probe = nengo.Probe(model.input.output)

with nengo.Simulator(model) as sim:  # Create a simulator
    sim.run(total_time)  # Run it for 1 second

plt.figure()
plt.plot(sim.trange(), sim.data[error_probe])
plt.title("Error")
plt.xlim(0, total_time)

plt.figure()
plt.plot(sim.trange(), spa.similarity(sim.data[input_probe], vocab))
plt.legend(vocab.keys(), fontsize='x-small')
plt.ylabel('State')

plt.figure()
plt.plot(sim.trange(), sim.data[output_probe])
plt.title("Output")
plt.xlim(0, total_time)
plt.show()
