# -*- coding: utf-8 -*-
import numpy as np


class Neuron: 
    """
    Class that defines a stochastic leaky-integrate and fire neuron which 
    tracks spiking history, time steps, number of steps and firing rates
    over a certain time bin 
    
    """
    
    def __init__(self,threshold, reset, flow=None, leakage=1, voltage=0, spiked=False, name = None):
        self.threshold = threshold
        self.flow = flow # flow neurons encode in WTA circuit
        self.reset = reset
        self.leakage = leakage
        self.spiked = spiked
        self.voltage = voltage
        self.nr_spikes = 0
        self.history = []
        self.timestep = 0
        self.pre_synapses = []
        self.post_synapses = []
        self.name = name
        self.is_valid = False # validity flag for encoder and decoder neurons
        
    def __str__(self):
        return "spikes: "+ str(self.nr_spikes)+ " "+ "voltage: "+ str(self.voltage) \
   + " " + "spiking history: "+ str(self.history)
   
    def noise(self):
       """
       add possion noise
       to neural integration process to
       kick off activation
       """
       if np.random.poisson(1,1) > 2:
           return 1
       return 0
   
    def spike_prob(self, voltage):
        return (1/20) * np.exp(voltage)
    
    def update(self):
        """
        Method that implements discretized difference equations 
        for updating LIF neurons
        """
     
        self.voltage *= self.leakage # multiplicative leakage
        if self.timestep < 21: # In the beginning only for delay == 0
                for synapse in self.pre_synapses:
                    if synapse.delay == 0:
                        self.voltage += synapse.pre.history[self.timestep - synapse.delay]*synapse.weight 
        else:
            for synapse in self.pre_synapses:
                 self.voltage += synapse.pre.history[self.timestep - synapse.delay]*synapse.weight 
        if self.flow != None: # transmitters add noise and bias
            self.voltage += 0.1*self.flow #+ self.noise()
        if self.voltage >= self.threshold:
        #if self.spike_prob(self.voltage) < np.random.rand():
            self.nr_spikes += 1
            self.history.append(1)
            self.reset_param()
        else:
            self.history.append(0)
        self.timestep += 1
        
     
    def reset_param(self):
        self.voltage = self.reset
        
    def add_synapse(self, synapse):
        self.pre_synapses.append(synapse)
    
    def firing_rate(self, window):
        """
        Estimation of p(r)
        over a certain time window (default 200ms)
        """
        rate = self.nr_spikes / window
        self.nr_spikes = 0 # restart count
        return rate
        
    def get_total_spikes(self):
        return self.nr_spikes
    
    
    def get_voltage(self):
        return self.voltage
    
class Synapse:
    """
    Class that defines a synapse of a neuron
    A synapse is formally a 4-tuple defined by (pre,post,weight, delay), consitute
    the presynaptic neuron, postsynaptic neuron the synaptic weigh and the
    synaptic delay adapated from Severra et al. 2016.
    """
    
    def __init__(self, pre, post, weight, delay):
        self.pre = pre
        self.post = post
        self.weight = weight
        self.delay = delay


class SpikingFlowNet:
    """
    wrapper class that keeps track
    of firing activity of neurons

    """
    def __init__(self, WTA_circuits, encoding_layers, decoding_layers, transmitters, window):
        self.WTA_circuits = WTA_circuits
        self.encoding_layers, self.decoding_layers = encoding_layers, decoding_layers
        self.transmitters = transmitters
        # set up recording for output neurons
        self.rates = dict()
        for name, neurons in WTA_circuits.items():
            self.rates[name] = dict((str(neuron.flow), []) \
                      for neuron in neurons if 'inhib' not in neuron.name and 'prio' not in neuron.name)
        self.sim_time = 0
        self.window = window
        
    def __str__(self):
        string = ""
       
        return string
    
    def sim(self, time):
        """
        Updates internal state of neurons in the network
        time : (msec)
        """
        self.sim_time = time # record simulation time for later use
        for t in range(time):
            # update transmitters
            for transmitter in self.transmitters:
                transmitter.update()
            # update WTA circuits 
            for name, neurons in self.WTA_circuits.items():
                for neuron in neurons:
                    neuron.update()
                    if t % self.window == 0 and 'inhib' not in neuron.name and 'prio' not in neuron.name: # update firing rate statistics after 200 ms
                        self.rates[name][str(neuron.flow)].append(neuron.firing_rate(self.window))
            # update encoding layers
            for name, encoding_layer in self.encoding_layers.items():
                [n_e.update() for n_e in encoding_layer] 
            # update decoding layers
            for name, decoding_layer in self.decoding_layers.items():
                [n_d.update()for n_d in decoding_layer] 
    
            
    def get_statistics(self):
        """
        Retrieve spiking statistics and flow
        in entire network
        """
       
        # decode flow in network
        total_flow = 0
        solution = [] # edge assignments
        for name, neurons in self.WTA_circuits.items():
            # retrieve WTA neuron with highest firing rate
            best = max([neuron for neuron in neurons if 'inhib'  \
                        not in neuron.name and 'prio' not in neuron.name], key=lambda n: self.rates[name][str(n.flow)][-1])
            if 'sink' in name:
                total_flow += best.flow
            solution += [(name, best.flow)]
        # get network activity over time
        A = np.zeros((len(self.WTA_circuits.keys()), int(self.sim_time / self.window)))
        for i, (name, neurons) in enumerate(self.WTA_circuits.items()):
            mean_wta = np.mean(np.array([self.rates[name][str(neuron.flow)] \
                                         for neuron in neurons if 'inhib' not in neuron.name and 'prio' not in neuron.name]), axis = 0)
            A[i,:] = mean_wta
        A = np.mean(A, axis = 0)
   
        for name, neurons in self.WTA_circuits.items():
            for neuron in neurons:
                if 'inhib' not in neuron.name and 'prio' not in neuron.name:
                    print(neuron.history[-30:-1])
            print("\n")
        return total_flow, self.rates, A, solution
    

        

