import nengo_spa as spa
import nengo

from constants import *

with spa.Network(label="Counter", seed=0) as model:
    model.q1 = spa.State(D)
