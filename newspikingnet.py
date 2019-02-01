import matplotlib.pyplot as plt
from collections import deque
from flownetwork import *
plt.style.use('ggplot')
from classes import SpikingFlowNet, Neuron, Synapse
        

    
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


def create_spiking_net(flow_net, max_flow, window):
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
           
    # construct connection pair for each connection
    seen = dict() # keep track of nodes that you have seen
    transmitters, flow_neurons, cap_neurons = [], [], []
    # create in and output neurons
    for name, capacity, priority in connections:
        pre, post = name.split('->')
        # pre neuron case
        if pre not in seen.keys():
            seen[pre] = 1
        else:
            seen[pre] += 1
        pre_neuron = Neuron(1,0, capacity, name = pre + '_' + '(' +name+ ')' + '_output', wta = capacity* (1/(2*priority)))  
        # post neuron case
        if post not in seen.keys():
            seen[post] = 1
        else:
            seen[post] += 1
         
        post_neuron = Neuron(1,0,capacity, name = post + '_' + '(' +name+ ')'+ '_input')
        
        if pre == 'source': # connect transmitter to source neuron
            transmitter = Neuron(1,1, capacity, name = 'transmitter')
            transmitter.voltage = 1
            pre_neuron.add_synapse(Synapse(transmitter, pre_neuron, 1, 1))
            transmitters.append(transmitter)

         
        cap_neuron = Neuron(capacity, capacity, 0, name = name)
        
        pre_neuron.add_synapse(Synapse(cap_neuron, pre_neuron, -1, 1))
        cap_neuron.add_synapse(Synapse(pre_neuron, cap_neuron, 1, 1))
        post_synapse = Synapse(pre_neuron, post_neuron, 1, 1)
        post_neuron.add_synapse(post_synapse)
        
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
    # create soft WTA circuit between output neurons of a group 
    node_cap_neurons = []
    for group in groups:
        
        in_cap = sum([cap.threshold for cap in cap_neurons if cap.name.split('->')[1] == group])
        out_cap = sum([cap.threshold for cap in cap_neurons if cap.name.split('->')[0] == group])
        group_cap = min(in_cap, out_cap)
  
       
        node_in_cap = Neuron(group_cap, 0, group_cap, name = group + '_in' +  '_cap')
        node_out_cap = Neuron(group_cap, 0, group_cap, name = group + '_out' +  '_cap')
            
        for neuron in flow_neurons:
            group_i, identifier, neuron_type = neuron.name.split('_')
            if group_i == group and neuron_type == 'input':
                neuron.add_synapse(Synapse(node_in_cap, neuron, -0.2*group_cap, 1))
                node_in_cap.add_synapse(Synapse(neuron, node_in_cap, 1, 1))
            if group_i == group and neuron_type == 'output':
                for neuron_j in flow_neurons:
                   group_j, identifier, neuron_typej = neuron.name.split('_') 
                   if group_j == group_i and neuron_typej == 'output':
                       neuron_j.add_synapse(Synapse(neuron, neuron_j, -neuron.wta*0.1, 1))
                       neuron.add_synapse(Synapse(neuron_j, neuron, -neuron_j.wta*0.1, 1))
              
        node_cap_neurons.append(node_in_cap)
        node_cap_neurons.append(node_out_cap)
    cap_neurons += node_cap_neurons
    # create spiking network
    spiking_net = SpikingFlowNet(flow_neurons, cap_neurons, transmitters, window)    
          
    return spiking_net
