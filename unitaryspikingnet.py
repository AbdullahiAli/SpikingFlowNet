import matplotlib.pyplot as plt
from collections import deque
from flownetwork import *
plt.style.use('ggplot')
from classes import SpikingFlowNet, Neuron, Synapse
import itertools      

    
def retrieve_all_connections(flow_net):
    """
    returns a list of connections with their
    corresponding capacities of the flow net
    """
    connections = []
    for node, neighbours in flow_net.items():
        for neighbour, capacity, priority in neighbours:
            connections += [(node + '->'+ neighbour, capacity, priority)]
    return connections
        

def create_connection(pre, post, weight, delay=20):
    """
    creates connnection between two neurons, with weigth
    1 and delay 20 ms
    """
    post.add_synapse(Synapse(pre, post, weight, delay))



def retrieve_in_connections(node, flow_net):
    """
    Retrieves incoming connections of node
    """
    in_connections = []
    for node_a, neighbours in flow_net.items():
        for neighbour, capacity, priority in neighbours:
            if neighbour == node:
                in_connections += [(node_a + '->'+ neighbour, capacity, priority)]
    return in_connections

def retrieve_out_connections(node, neighbours):
    """
    Retrieves outgoing connections of node
    """
    return [(node + '->'+ neighbour, capacity, priority) \
            for neighbour, capacity, priority in neighbours]
    
    


def connect_input_layer(n_e, ne_neg, in_connections, WTA_circuits):
    """
    Connects input WTA circuits to encoding neuron n_e
    """
    for name, capacity, priority in in_connections:
        WTA = WTA_circuits[name]
        for neuron in WTA:
            if 'inhib' not in neuron.name and 'prio' not in neuron.name:
                if (neuron.flow, name) in n_e.name: # check identifier and flow are in combination
                    create_connection(neuron, n_e, 1)
                    create_connection(n_e, neuron, 1) # positive feedback loop
                    create_connection(ne_neg, neuron, -1) # penalize for invalid flow
                else:
                    create_connection(neuron, n_e, -1) # negative constraint
                    create_connection(n_e, neuron, -1) # negative feedback loop

def connect_encode_decode(encoding_layer, decoding_layer):
    """
    Connects encoding neurons n_e to decoding neuron n_d
    that represents the same flow. If there is no 
    n_d that represents the same flow it means that the flow n_e
    represents is too high. In that case, n_e is connected to
    the n_d that represents the highest flow and inhibitory
    decoder neuron nd_neg is connected to inhibitory encoder
    ne_neg to penalize the invalid flow.
    """  
    for n_e, ne_neg in encoding_layer:
        best_neuron = None
        best_neg  = None
        best_flow = float('-inf')
        is_connected = False
        for n_d, nd_neg in decoding_layer:
            if n_d.flow > best_flow:
                best_neuron = n_d
                best_neg = nd_neg
                best_flow = n_d.flow
            if n_d.flow == n_e.flow: # found matching decoder neuron
                create_connection(n_e, n_d, 1) # feedforward excitation
                create_connection(n_d, n_e, 1) # positive feedback
                is_connected = True
        if not is_connected:
            #create_connection(n_e, best_neuron, 1)
            create_connection(n_e, best_neg, 1) # penalize for invalid flow
            create_connection(best_neg, ne_neg, -1)
            #create_connection(n_e, n_e, -1) # penalize for invalid flow]
            
#    # lateral inhibition for encoding and decoding layers
#    for n_d, nd_neg in decoding_layer:
#        for n_de, _ in decoding_layer:
#            if n_d.flow != n_de.flow:
#                create_connection(n_d, n_de, -1) # inhibition if not coding for same flow
#    for n_e, ne_neg in encoding_layer:
#        for n_ed, _ in encoding_layer:
#            if n_e.flow != n_ed.flow:
#                create_connection(n_e, n_ed, -1) # inhibition if not coding for same flow
        
def connect_output_layer(n_d, nd_neg, out_connections, WTA_circuits):
    """
    Connects output WTA circuits to decoding neurons n_d.
    Each n_d represents a specific combination flows of the 
    outgoing arcs. If a neuron in the WTA circuit represents 
    one of the outgoing flows in n_d then, n_d will connect to
    that neuron. 
    """
    for name, capacity, priority in out_connections:
        # retrieve WTA circuits
        WTA = WTA_circuits[name]
        # flow of decoder neuron represents 
        # combination of outgoing neurons
        # Connect decoder to neurons in WTA circuit
        # that are part of the flow combination
        for neuron in WTA:
            if (neuron.flow, name) in n_d.name: # check identifier and flow are in combination
                create_connection(n_d, neuron, 1)
                create_connection(neuron, n_d, 1) # positive feedback loop
            else:
                create_connection(n_d, neuron, -1) # negative constraint
                    
def set_up_wta(WTA, priority):
    """
    Function that connects WTA neurons in order
    to set up competition and adds bias priority neuron
    """
    # create auxilary neurons
    # threshold of 2 so that it only fires if atleast two neurons are active
    inhib = Neuron(1, 0, name='inhib')
    prio = Neuron(1, 1, name='prio') # encodes priority bias in competition
    #prio.voltage = 1
    for neuron in WTA:
        #create_connection(neuron, inhib, 1)
        #if 'source' in neuron.name:
            #create_connection(inhib, neuron, -1)
        #else:
            #create_connection(inhib, neuron, -1)
        #create_connection(neuron, prio, 1)
        # create lateral inhibition between neurons in WTA
        for neuron_j in WTA:
            if neuron_j.flow != neuron.flow:
                create_connection(neuron, neuron_j, -1)
        #create_connection(prio, neuron, (1/(10*priority))*neuron.flow)
            
    WTA += [inhib, prio]
    return WTA


def create_spiking_net(flow_net, window):
    """
    function that maps a flow network
    into a spiking network using a unitary representation
    of flow.
    For each arc we create a WTA circuit with c neurons
    where c is the capacity of the neurons. 
    For each WTA circuit we create a bias neurons
    that inputs a bias current based on the (ordered) priority of
    the edge. This is encoded in such a way that an edge that is closer
    to the sink will reach it's full capacity earlier than an edge that is
    further away from the sink 
    
    For each node we retrieve the incoming and outgoing WTA circuits
    We create an encoding layer that combines all possible flow
    from the incoming WTA circuits and connect the incoming WTA neurons
    to these encoding neurons acordingly. At most one of these encoding
    neurons can be active.
    Next we connect the enconding neurons
    to the outgoing WTA neurons in the following way:
        - Encoding neuron i with flow f_i will connect to all outgoing
        WTA neurons which code for a flow <= f_i. In this way incoming and
        outgoing flow is preserved. 
    """
    # lable connections in flow net according to distance to sink
    labeled_flow_net = label_flow_net(flow_net)
    # retrieve connections
    connections = retrieve_all_connections(labeled_flow_net)
    WTA_circuits = dict() 
    transmitters = []
    # create WTA circuits
    for name, capacity, priority in connections:
   
        # create WTA circuit
        WTA = [Neuron(1, 0, i, name=name + " " + str(i)) for i in range(capacity + 1)]
      
        # set up connectivity of WTA
        WTA = set_up_wta(WTA, priority)
        if 'source' in name:
            # create transmitter to start up network
            n_t = Neuron(1,1, voltage=1)
            for neuron in WTA:
                 if 'inhib' not in neuron.name and 'prio' not in neuron.name:
                    create_connection(n_t, neuron, 1)
        transmitters.append(n_t)
        # add WTA circuit to dictionary
        WTA_circuits[name] = WTA
        
    # set up encoding and decoding layers
    encoding_layers, decoding_layers = dict(), dict()
    for node, neighbours in labeled_flow_net.items():
        encoding_layer, decoding_layer = [], []
        # source and sink do not require encoding layers
        if node != 'source' and node != 'sink':
             # incoming connections
             in_connections = retrieve_in_connections(node, labeled_flow_net)
             out_connections = retrieve_out_connections(node, neighbours)
             in_flows, out_flows = [], []
             # retrieve possible in_flows and link to originating connection
             for name, capacity, priority in in_connections:
                 flow_range = list(range(0, capacity + 1)) # range of flow in WTA
                 in_flows.append([(flow,name) for flow in flow_range]) # add identifier
                 
            # retrieve possible out_flows and link to originating connection
             for name, capacity, priority in out_connections:
                 flow_range = list(range(0, capacity + 1)) # range of flow in WTA
                 out_flows.append([(flow, name) for flow in flow_range])
             
             # create encoding layer
             in_combinations = list(itertools.product(*in_flows))
             for combination in in_combinations:
                 n_e = Neuron(len(combination), 0, name = combination, flow = sum([flow for flow, name in combination]))
                 ne_neg = Neuron(1, 0, name = combination)
                 connect_input_layer(n_e, ne_neg,  in_connections, WTA_circuits)
                 encoding_layer.append((n_e, ne_neg))
             # create decoding layer
             out_combinations = list(itertools.product(*out_flows))
             for combination in out_combinations:
                 n_d = Neuron(1, 0, name = combination, flow = sum([flow for flow, name in combination]))
                 nd_neg = Neuron(1, 0, name = combination)
                 connect_output_layer(n_d, nd_neg, out_connections, WTA_circuits)
                 decoding_layer.append((n_d, nd_neg))
        connect_encode_decode(encoding_layer, decoding_layer)
        encoding_layers[node] = encoding_layer
        decoding_layers[node] = decoding_layer
    # set up network
    spiking_net = SpikingFlowNet(WTA_circuits, encoding_layers, decoding_layers, transmitters, window)
          
    return spiking_net


