import nengo
import nengo_spa as spa

import numpy as np

dimensions = 16

vocab = spa.Vocabulary(dimensions)
vocab.populate("A; B; C")

with spa.Network() as model:
    model.input = spa.Transcode(output_vocab=vocab)
    model.state = spa.State(vocab, subdimensions=dimensions, represent_identity=False)
    assert len(model.state.all_networks) == 1
    state_net = model.state.all_networks[0]

    model.bg = spa.BasalGanglia(action_count=3)
    model.thal = spa.Thalamus(action_count=3)

    nengo.Connection(model.input.output, model.state.input, synapse=None)

    # TODO: Connect from the neurons for the learned connection somehow
    # conditions
    nengo.Connection(model.state.output, model.bg.input[0],
                     transform=[vocab["A"].v])
    nengo.Connection(model.state.output, model.bg.input[1],
                     transform=[vocab["B"].v])
                     #transform=np.zeros((1, dimensions)))
    nengo.Connection(model.state.output, model.bg.input[2],
                     transform=[vocab["C"].v])

    nengo.Connection(model.bg.output, model.thal.input)

    # action results
    nengo.Connection(model.thal.output[0], model.state.input,
                     transform=vocab["B"].v.reshape(-1, 1))
    nengo.Connection(model.thal.output[1], model.state.input,
                     transform=vocab["C"].v.reshape(-1, 1))
    nengo.Connection(model.thal.output[2], model.state.input,
                     transform=vocab["A"].v.reshape(-1, 1))

    # TODO: get the error from the expected state somehow
    # TODO: use that error to learn the connections somehow
