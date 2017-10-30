import nengo_spa as spa


class Cleanup(spa.WTAAssocMem):

    def __init__(self, vocab):
        super().__init__(input_vocab=vocab, threshold=0.3, function=lambda x: x > 0.)


class HeteroMap(spa.WTAAssocMem):

    def __init__(self, vocab, input_keys, output_keys):
        super().__init__(input_vocab=vocab, threshold=0.3, mapping=dict(zip(input_keys, output_keys)),
                         function=lambda x: x > 0.)
