import nengo
import nengo_spa as spa
import nengo_gui

import numpy as np

dimensions = 16

vocab = spa.Vocabulary(dimensions)
vocab.populate("A; B; C")


def objective_func(t, x):
    return x[0] - np.dot(x[1:], vocab["B"].v)


with spa.Network() as model:
    model.input = spa.Transcode(output_vocab=vocab)
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
