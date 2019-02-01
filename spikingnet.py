import matplotlib.pyplot as plt
from collections import deque
from flownetwork import *
plt.style.use('ggplot')

class SpikingFlowNet:
    """
    wrapper class that consists of four groups of neurons
    a delay clock that determines the time, a transmitter
    that transmits the time to the flow neurons and an input neuron
    """
    def __init__(self, delay_clock, transmitters, flow_neurons, capacity_neurons, input_neuron, decay_neurons):
        self.delay_clock = delay_clock
        self.transmitters = transmitters
        self.flow_neurons, self.capacity_neurons = flow_neurons, capacity_neurons
        self.input_neuron = input_neuron
        self.sink = [neuron for neuron in self.flow_neurons  if 'sink' in neuron.name ][0]
        self.decay_neurons = decay_neurons
        
    def __str__(self):
        string = "Clock neurons:\n"
        for neuron in self.delay_clock:
            string += str(neuron) +  "\n" 
        string += "\n"
        string += "Transmitters:\n"
        for neuron in self.transmitters:
            string += str(neuron) + "\n"
        string += "Flow neurons:\n"
        for neuron in self.flow_neurons:
            string += neuron.name + " " +str(neuron) + "\n"
        string += "\n"
        string+= "Decay neurons: \n"
        for neuron in self.decay_neurons:
            string += neuron.name + " " +str(neuron) + "\n"
        string += "\n"
        string += "Capacity neurons:\n"
        for neuron in self.capacity_neurons:
            string += neuron.name + " " + str(neuron) + "\n"
        return string
    
    def update(self):
        """
        Updates internal state of neurons in the network
        """
        # update input neuron
        #self.input_neuron.update()
        # Update clock neurons
        for clock_neuron in self.delay_clock:
            clock_neuron.update()
        # transmit the time
        for transmitter in self.transmitters:
            transmitter.update()
        # update flow neurons
        for flow_neuron in self.flow_neurons:
            flow_neuron.update()
        # update capacity neurons
        for capacity_neuron in self.capacity_neurons:
            capacity_neuron.update()  
        # update decay neurons
        for decay_neuron in self.decay_neurons:
            decay_neuron.update()
            
    def get_statistics(self):
        """
        Clocks spiking statistics and flow in network
        """
        # retrieve total number of spikes in network
        all_neurons = self.delay_clock + self.transmitters + self.flow_neurons + self.capacity_neurons + self.decay_neurons
        total_spikes = sum([neuron.get_total_spikes() for neuron in all_neurons])
        # decode flow in network
        total_flow = sum([neuron.nr_spikes for neuron in self.flow_neurons if '->sink)_input' in neuron.name])
        return total_spikes, total_flow, self.sink.timestep  + 1
    
    def has_terminated(self):
        """
        Check if all decay neurons have
        decayed away
        """
        return all([decay_neuron.has_decayed() for decay_neuron in self.decay_neurons])
        
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



class Neuron: 
    """
    Class that defines a leaky-integrate and fire neuron which 
    also tracks spiking history, time steps and the number of spikes.
    In this class a neuron is formally a 3-tuple of (threshold, reset, leakage)
    which consitute the firing threshold, reset potential and multiplicative
    membrame leakage adapated from Severra et al. 2016.
    """
    
    def __init__(self,threshold, reset, leakage=1, voltage=0, spiked=False, name = None, decay=False):
        self.threshold = threshold
        self.reset = reset
        self.leakage = leakage
        self.spiked = spiked
        self.spike_time = None
        self.voltage = voltage
        self.nr_spikes = 0
        self.history = []
        self.timestep = 0
        self.pre_synapses = []
        self.post_synapses = []
        self.name = name
        self.decay = decay
        self.decayed = False
    def __str__(self):
        return "spikes: "+ str(self.nr_spikes)+ " "+ "voltage: "+ str(self.voltage) \
   + " " + "spiking history: "+ str(self.history)
    
    def update(self):
        """
        Method that implements discretized difference equations 
        for updating LIF neurons
        """
     
        self.voltage *= self.leakage # multiplicative leakage
        # If neuron is a decay neuron, check if decay is completed 
        if self.decay == True: 
            if self.voltage == 0:
                self.decayed = True
            else:
                self.voltage -= 1 # decrement by one 
        
        if self.timestep == 0: # In the beginning only for delay == 0
                for synapse in self.pre_synapses:
                    if synapse.delay == 0 and self.decayed == False:
                        self.voltage += synapse.pre.history[self.timestep - synapse.delay]*synapse.weight
        elif self.decayed == False:
            for synapse in self.pre_synapses:
                self.voltage += synapse.pre.history[self.timestep - synapse.delay]*synapse.weight
                
        if self.voltage >= self.threshold:
            if self.spiked == False: # get time of first spike
                self.spiked = True
                self.spike_time = self.timestep
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
    
    def add_post_synapse(self, synapse):
        self.post_synapses.append(synapse)
        
    def get_total_spikes(self):
        return self.nr_spikes
    
    def has_spiked(self):
        return self.spiked
    
    def has_decayed(self):
        return self.decayed
    
    def get_spike_time(self):
        return self.spike_time + 1 # add one to be in line with counting from 1
    
    def get_voltage(self):
        return self.voltage
        
class MultiplexNeuron(Neuron):
    """
    Special neuron class for output neurons in order
    to deal with spikes from multiple input neurons.
    This neuron implements time division multiplexing.
    If two spikes arrive at the same time, the neuron will
    order these two spikes in time by having a reset voltage 
    of self.voltage -1.
    """
    def __init__(self, threshold, reset, leakage=1, voltage=0, spiked=False, name = None, decay=False):
        super().__init__(threshold, reset, leakage, voltage, spiked, name, decay)
        
    def reset_param(self):
        
        self.voltage -= 1
        
        
def retrieve_connections(flow_net):
    """
    returns a list of connections with their
    corresponding capacities of the flow net
    """
    connections = []
    for node, neighbours in flow_net.items():
        for neighbour, capacity, priority in neighbours:
            connections += [(node + '->'+ neighbour, capacity, priority)]
    return connections
        

def create_connection(pre, post, weight, delay=1, neurons=None, version2=False):
    """
    creates connnection between two neurons, with weigth
    1 and delay 1
    """
    if version2:
        post.add_synapse(Synapse(pre, post, weight, delay))
    else:
        pre_neuron, post_neuron = None, None
        for neuron in neurons:
            if neuron.name == pre:
                pre_neuron = neuron
            if neuron.name == post:
                post_neuron = neuron
        post_neuron.add_synapse(Synapse(pre_neuron, post_neuron, weight, delay))


def get_total_capacity(node_neighbours):
    """
    Find the total capacity of a node by summing
    the capacity of the outgoing edges of the node
    """
    return sum([cap for node, cap in node_neighbours])

def in_connections(spiking_net, pre, post):
    """
    Checks if pre is connected to post neuron
    """
    neurons = spiking_net.flow_neurons
    for neuron in neurons:
        if neuron.name == pre:
            for neuronj in neurons:
                if neuronj.name == post and neuron in neuronj.pre_synapses:
                    return True
    return False


def create_spiking_net(flow_net, max_flow):
    """
    Transforms a flow net into a spiking net
    using the custom classes defined above
    
    For each connection (a,b) with capacity c in the network
    we create output neuron a_i_output (resembling the node that sends flow),
    input neuron b_i_input (resembling the node that receives flow) and a 
    capacity neuron c_i that keeps track of the flow through this connection.
    
    Naming: flow neurons are named in the form group_identifier_type
    group: node in the flow network it belongs to (e.g. source, a, b, sink)
    identifier: the index of the node in the group (integer)
    type: wheter it receives input or sends output (input or output)
    
    Within a group, all input neurons send their flow to output neurons that
    then send their flow to input neurons in designated groups. In this way the
    topology of the flow network will be preserved while 
    seperately modeling the individual connections.
    
    The excitability of an output neuron a_i is determined
    by the priority of the edge (a,b) in the flow network, which
    is defined as 1 + the shortest path from b to the sink. 
    In this way the spiking networks is biased towards pushing
    spikes through the shortest augmenting path, analogous
    to the Edmond Karps implementation of the Ford-Fulkerson Method.
    """
    # lable connections in flow net according to distance to sink
    labeled_flow_net = label_flow_net(flow_net)
    # retrieve connections
    connections = retrieve_connections(labeled_flow_net)
    
    # create clock mechanism
    delay_clock = [Neuron(1,1, voltage=1)]
    transmitters = []
    feedback_flow = get_total_capacity(flow_net['source']) # upper bound on flow
    
            
    # construct connection pair for each connection
    seen = dict() # keep track of nodes that you have seen
    flow_neurons, cap_neurons, decay_neurons = [], [], []
    input_neuron = Neuron(1,1, voltage=1, name = 'input') # legacy code
    # create in and output neurons
    for name, capacity, priority in connections:
        pre, post = name.split('->')
        # pre neuron case
        if pre not in seen.keys():
            seen[pre] = 1
        else:
            seen[pre] += 1
        if pre != 'source':
            pre_neuron = MultiplexNeuron(1, 0, name = pre + '_' + '(' +name+ ')' + '_output')  
        else:
            pre_neuron = Neuron(1,0, name = pre + '_' + '(' +name+ ')' + '_output')  
        # post neuron case
        if post not in seen.keys():
            seen[post] = 1
        else:
            seen[post] += 1
        # if post neuron is a sink create additional decay neuron for termination purposes
        if post == 'sink':
             post_neuron = Neuron(1, 0, name = post + '_' + '(' +name+ ')'+ '_input')
             decay_neuron = Neuron(2*capacity, 0, name = post+ '_' + '(' +name+ ')' +  '_decay', decay=True)
             decay_neuron.voltage = 2*capacity
             decay_neuron.add_synapse(Synapse(post_neuron, decay_neuron, 1 , 1))
             decay_neurons.append(decay_neuron)
        else:
            post_neuron = Neuron(1,0, name = post + '_' + '(' +name+ ')'+ '_input')
        
        if pre == 'source': # connect transmitter to source neuron
            transmitter = Neuron(1,0)
            # connect transmitter neuron to scheduler
            transmitter.add_synapse(Synapse(delay_clock[0], transmitter, 1, 1))
            pre_neuron.add_synapse(Synapse(transmitter, pre_neuron, 1,1)) # excitatory connection
            cap_neuron = Neuron(capacity - 1, capacity - 1, name = name)
            transmitter.add_synapse(Synapse(cap_neuron, transmitter, -1, 1))
            cap_neuron.add_synapse(Synapse(transmitter, cap_neuron, 1, 1))
            transmitters.append(transmitter)
        else:
            if capacity == 1:
                cap_neuron = Neuron(capacity, capacity, name = name)
            else:
                cap_neuron = Neuron(capacity - 1, capacity - 1, name = name)
            pre_neuron.add_synapse(Synapse(cap_neuron, pre_neuron, -capacity , 1))
            cap_neuron.add_synapse(Synapse(pre_neuron, cap_neuron, 1, 1))
        post_synapse = Synapse(pre_neuron, post_neuron, 1, 1)
        post_neuron.add_synapse(post_synapse)
        pre_neuron.add_post_synapse(post_synapse)
        
        flow_neurons.append(pre_neuron)
        flow_neurons.append(post_neuron)
        cap_neurons.append(cap_neuron)
        
        
    # Connect flow groups
    groups = [] # bookkeeping for later
    for neuron_i in flow_neurons:
        group_i, identifier_i, neuron_type_i = neuron_i.name.split('_')
        if group_i not in groups and group_i not in ['source', 'sink']:
            groups.append(group_i)
        if neuron_type_i == 'input': # connect it to all input neurons in the group
            for neuron_j in flow_neurons:
                group_j, identifier_j, neuron_type_j = neuron_j.name.split('_')
                if group_j == group_i and neuron_type_j == 'output':
                    neuron_j.add_synapse(Synapse(neuron_i, neuron_j, 1, 1)) # excitatory connection
                    
    # create capacity neuron to keep track of entire flow through a node
    node_cap_neurons = []
    for group in groups:
        #in_cap_neurons = [cap for cap in cap_neurons if cap.name.split('->')[1] == group]
        #out_cap_neurons = [cap for cap in cap_neurons if cap.name.split('->')[0] == group]
        in_cap = sum([cap.threshold + 1 for cap in cap_neurons if cap.name.split('->')[1] == group])
        out_cap = sum([cap.threshold + 1 for cap in cap_neurons if cap.name.split('->')[0] == group])
        group_cap = min(in_cap, out_cap)
        #nr_output_neurons = len([n for n in flow_neurons if group in n.name.split('_') and 'output' in n.name.split('_')])
        #nr_input_neurons = len([n for n in flow_neurons if group in n.name.split('_') and 'input' in n.name.split('_')])
       
        node_in_cap = Neuron(group_cap - 1,group_cap - 1, name = group + '_in' +  '_cap')
        node_out_cap = Neuron(group_cap - 1, group_cap - 1, name = group + '_out' +  '_cap')
        #print('Node: ' + node_cap.name + '\n' + 'capacity: ' + str(node_cap.threshold))
        #for neuron in flow_neurons: # connect node_in_cap as feedback signal to all groups projecting to this group
            #group_i, identifier, neuron_type = neuron.name.split('_')
            #if group in identifier and neuron_type == 'output':
                #neuron.add_synapse(Synapse(node_in_cap, neuron, -group_cap, 1))
            
        for neuron in flow_neurons:
            group_i, identifier, neuron_type = neuron.name.split('_')
            if group_i == group and neuron_type == 'input':
                neuron.add_synapse(Synapse(node_in_cap, neuron, -group_cap, 1))
                node_in_cap.add_synapse(Synapse(neuron, node_in_cap, 1, 1))
            if group_i == group and neuron_type == 'output':
                cap_neuron = [cap for cap in cap_neurons if cap.name in identifier][0]
                capacity = cap_neuron.threshold - 1
                neuron.add_synapse(Synapse(node_out_cap, neuron, -group_cap, 1))
                node_out_cap.add_synapse(Synapse(neuron, node_out_cap, 1, 1))
              
                #max_spikes = cap_neuron.threshold + 1
                #print('max_spikes: ' + str(max_spikes))
                #neuron.threshold =  math.ceil(group_cap /max_spikes)
                #print('threshold: ' + str(neuron.threshold) + '\n')
        node_cap_neurons.append(node_in_cap)
        node_cap_neurons.append(node_out_cap)
    cap_neurons += node_cap_neurons
    # create spiking network
    spiking_flow_net = SpikingFlowNet(delay_clock, transmitters, flow_neurons, cap_neurons, input_neuron, decay_neurons)    
          
    return spiking_flow_net
