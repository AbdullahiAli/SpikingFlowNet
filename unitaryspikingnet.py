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
    1 and default delay 20 ms
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
    
    


def create_spiking_search_net(flow_net):
    """
    Function that maps a flow network into 
    a spiking network in which we can find the shortest
    path from source to sink on the basis of the First spike time of
    neurons (FST).
    """
    # create transmitter to kickstart network
    transmitter = SearchNeuron('transmitter', 1, 0)
    transmitter.voltage = 1
    flow_neurons = []
    # go through each connection and create neuron
    for node in flow_net.keys():
        for n, c, f, rf in flow_net[node]:
            if c > f: # we can still push flow through this connection
                flow_neuron = SearchNeuron(node + '->' + n, 1, 0)
            else: # We cannot push flow, so prevent this neuron from firing
                flow_neuron = SearchNeuron(node + '->' + n, float('inf'), 0)
            flow_neurons.append(flow_neuron)
        # connect neurons 
        for neuron in flow_neurons:
            if 'sink' in neuron.identifier: # connect transmitter to sink 
                create_connection(transmitter, neuron, 1, delay=1)
            source, dest = neuron.identifier.split('->')
            for neuron_j in flow_neurons:
                source_j, dest_j = neuron_j.identifier.split('->')
                if dest_j == source: # we found a match
                    create_connection(neuron, neuron_j, 1, delay=1)
                 
    return SpikingSearchNet(flow_neurons, transmitter)