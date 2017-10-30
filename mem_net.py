import nengo_spa as spa
from nengo.networks.workingmemory import InputGatedMemory as WM


class MemNet(spa.Network):
    def __init__(self, mem_vocab, label=None):
        super().__init__(label)

        with self:
            self.mem = WM(100, mem_vocab.dimensions, difference_gain=15)
            self.mem.label = "mem"

        self.gate = self.mem.gate
        self.input = self.mem.input
        self.output = self.mem.output

        self.declare_input(self.gate, None)
        self.declare_input(self.input, mem_vocab)
        self.declare_output(self.output, mem_vocab)
