# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
plt.style.use('ggplot')

start_scope()
weight = 1 # default weight param
tau = 10*ms # default time constant
sigma = 0.1
eqs = 'dv/dt = -v/tau + sigma*xi*tau**-0.5 : volt'

# Flow neurons
S1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
S2 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
A_i1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
A_i2 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
A_o1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')

B_i1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
B_o1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
B_o2 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
T1 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')
T2 = NeuronGroup(10, eqs, threshold='v>1', reset='v=0')

# Capacity neurons
A_c = NeuronGroup(10, eqs, threshold='v>3', reset='v=0')
B_c = NeuronGroup(10, eqs, threshold='v>3', reset='v=0')
# Flow connections
S_A = Synapses(S1, A_i1, on_pre='v += weight')
S_B = Synapses(S2, B_i1, on_pre='v += weight')
B_A = Synapses(B_o1, A_i2,  on_pre='v += weight')
A_T = Synapses(A_o1, T1, on_pre='v+=weight')
B_T = Synapses(B_o2, T2, on_pre='v+=weight')

# IO connections
Ai1_o1 = Synapses(A_i1, A_o1, on_pre='v+=weight')
Ai2_o1 = Synapses(A_i2, A_o1, on_pre='v+=weight')
Bi1_o1 = Synapses(B_i1, B_o1, on_pre='v+=weight')
Bi1_o2 = Synapses(B_i1, B_o2, on_pre='v+=weight')

# Capacity Connections
S_A.connect()
S_B.connect()
M = SpikeMonitor(B_i1)
# Now we can just run once with no loop
run(1*second)
plot(M.t/ms, M.i, '.')
xlabel(r'$\tau$ (ms)')
ylabel('Firing rate (sp/s)');