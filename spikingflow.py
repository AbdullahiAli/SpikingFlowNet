#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Most recent version: Februari 6 2019
@author: Abdullahi Ali email: a.ali@student.ru.nl
--------
This script generates and tests spiking network implementations 
of flow networks and computes the maximum flow in a given network.
This script contains all the necessary classes methods and helper functions
to generate a flow network, transform it into spiking net and to evaluate the energy 
demands and accuracy of the spiking network

Usage:
To obtain a quick idea of this file you can run test() which transforms
an example flow network into a spiking network and displays some spike statistics 

If you want to run the actual experiment, you have call experiment()

Both calls can be found at the bottom of the script. Simply uncomment the 
call you're interested in.

Lines ~30 - ~240 helper functions and code to generate flow networks and
their reference max flow computed with the Edmonds Karp algorithm

Lines ~240 - ~380 Neuron, Synapse and Network classes

Lines ~380 - end of script, actual SpikeFlow algorithm and helper functions 
to generate the result. 

See the function definitions for a detailed description of their functionality.

Known problems:
Spiking net can sometimes run indefinitely when calling experiment()
Abort the script if takes more than 2 minutes and restart.
"""

import matplotlib.pyplot as plt
from collections import deque
from unitaryspikingnet import *
from flownetwork import *
import numpy as np
    

SIM_TIME = 20000 # 5s simulation
WINDOW = 200 # time window for spike bins (in msec)
def spike_flow(flow_network):
    """
    Algorithm that computes the maximum flow
    in a flow network by converting it to a spiking 
    network
    """
    # create spiking flow network
    spiking_network = create_spiking_net(flow_network, WINDOW)
    # simulate for 1 second
    spiking_network.sim(SIM_TIME)
    return spiking_network

def Spiking_EK(gen_flow_net):
    """
    Calculates Spiking E-K
    """
    """
    Initialize flow and flow net and boolean indicator
    """
    flow_net, max_flow, augmenting_path = dict(), 0, True
    
    """
    Create graph with edges that consitute of 4-tuples
    (neighbour, capacity, flow, reversed flow)
    """
    for node in gen_flow_net:
        flow_net[node] = [(neighbour,capacity, 0,0) for neighbour, capacity in gen_flow_net[node]]
    av_spikes, av_time = [], [] # keep track of average number of spikes and average number of time steps
    while augmenting_path: # find augmenting path using spiking net
        spiking_net = create_spiking_search_net(flow_net)
        # simulate spiking net for 100 timesteps
        while not spiking_net.terminated():
            spiking_net.sim(1)
        spikes, time = spiking_net.get_statistics()
        av_spikes.append(spikes)
        av_time.append(time)
        # get data
        neurons = spiking_net.get_data()
        # sort neurons on FST
        neurons.sort(key=lambda x: x[1], reverse=True)
        path = []
        nodes  = [] # keep track of all nodes on path
        # construct  shortest path
        if len(neurons) != 0:
            start, FST = neurons[0][0], neurons[0][1]
            path  = [(start, FST)]
            for curr, FST in neurons:
                prev, old_FST = path[-1] # retrieve most recent end on path
                dest = prev.split('->')[1]
                source = curr.split('->')[0]
                nodes.append(source)
                nodes.append(dest)
                # check if dest of prev and source of curr match
                if  dest == source and FST < old_FST: # valid continuation of path
                    path.append((curr, FST))
        # preprocessing of path to fall in line with previous E-K implementation (artifact)
        pred = dict()
        path = [name for name, FST in path]

        for name in path:
            node, neighbour = name.split('->')
            for n, c, f, rf in flow_net[node]:
                if n == neighbour:
                    pred[neighbour] = (node, neighbour, c, f, rf)
        if 'sink' in pred.keys() and 'source' in nodes: 
            # we found an augmenting path see how much flow we can push
            dflow = float("inf") # change in flow
            node = 'sink'
            reached = False
           
            while not reached:
                source, sink, capacity, flow, rev_flow = pred[node]
                dflow = min(dflow, capacity - flow)
                node = source
                if node == 'source':
                    reached = True
                
            # and update the edges with this flow
            node = 'sink'
            reached = False
           
            while not reached:
                source, sink, capacity, flow, rev_flow = pred[node]
                for i, edge in enumerate(flow_net[source]):
                        neighbour, capacity, flow, rev_flow = edge
                        if neighbour == sink:
                            # update flow and reversed flow
                            flow_net[source][i] = (neighbour, capacity, flow + dflow, rev_flow - dflow)
                node = source
                if node == 'source':
                    reached = True
                    
            max_flow += dflow
        else:
            # if no augmenting path is found terminate algorithm and return found max flow
            augmenting_path = False
               
    return max_flow, np.mean(av_spikes), np.mean(av_time)

def calculate_spike_stats(networks):
    """
    Calculates spiking net statistics
    for a given set of networks
    """
    spikes, flow, error, time = [], [], [], []
    for flow_net, max_flow in networks:
        net_flow, net_spikes, net_time = Spiking_EK(flow_net)
        net_error = abs(max_flow - net_flow) / max_flow
        spikes.append(net_spikes)
        flow.append(net_flow)
        error.append(net_error)
        time.append(net_time)
    avg_error =  sum(error)/len(networks)
    avg_spikes =  sum(spikes)/len(networks)
    avg_time = sum(time)/ len(networks)
 
    return avg_error, avg_spikes, avg_time

def experiment():
    """
    Main function that generates the experiment
    We ran over network sizes 
    """
    
    total_error, total_spikes, total_time = [], [], []
    small, large = 5, 40
    for i in range(small, large):
        networks = [generate_flow_net(i,i-1,10) for j in range(40)]
        error, spikes, time = calculate_spike_stats(networks)
        total_error.append(error)
        total_spikes.append(spikes)
        total_time.append(time)
    
    # plot stats
    plt.figure(figsize=(15,6))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9,  top= 0.9, wspace=0.3, hspace=0.2)
    plt.subplot(1,3,1)
    plt.plot(range(small,large), total_spikes)
    plt.title("Energy demands")
    plt.xlabel('network size')
    plt.ylabel('number of spikes (averaged)')
    plt.subplot(1,3,2)
    plt.plot(range(small,large), total_error)
    plt.title("Error between E-K and spiking net")
    plt.xlabel('network size')
    plt.ylabel('error (averaged)')
    plt.subplot(1,3,3)
    plt.plot(range(small,large), total_time)
    plt.title("Total time")
    plt.xlabel('network size')
    plt.ylabel('number of time steps (averaged)')
    plt.show()

def run_network(flow_net):
    """
    Helper function that runs a spiking
    net simulation for a specific network flow
    graph
    """
    print("E - K flow: " + str(Edmonds_Karp(flow_net)))
    print("spiking E- K flow: " + str(Spiking_EK(flow_net)))
    
def run_network_old(flow_net):
    """
    Helper function that runs a spiking
    net simulation for a specific network flow
    graph
    """
    spiking_flow_net = spike_flow(flow_net)
    flow, rates, A, solution= spiking_flow_net.get_statistics()
    print("E - K flow: " + str(Edmonds_Karp(flow_net)))
    print("spiking net flow: " + str(flow))
    times = np.linspace(0,SIM_TIME, SIM_TIME/WINDOW) # generate time vector
    plt.figure(1, figsize = (7,7))
    plt.title('Firing rates of neurons')
    plt.xlabel('Time (msec)')
    plt.ylabel('Firing rate density')
    for key, neurons in rates.items():
        for neuron, rate in neurons.items():
            plt.plot(times, rate)
            plt.legend(key + neuron)
    plt.show()
    
    plt.figure(2, figsize= (7,7))
    plt.title('Network activity A(t)')
    plt.xlabel('Time (msec)')
    plt.ylabel('Activity (normalized over neuron activity)')
    plt.plot(times, A)
    plt.show()
    print("Firing rates for wta neurons: " + '\n')
    for key, neurons in rates.items():
        for neuron, rate in neurons.items():
            print('edge: ' + key + ' ' + 'flow: ' \
                  + neuron + ' ' + 'Firing rate : ' + str(rate[-1]))
    print('\n')
    print("Solution: " + '\n')
    for name, flow in  solution:
        print(name + ":" + str(flow))
        
def test0():
    """
    test to determine if spiking flow
    net algorithms works correct
    """
    # example flow net 
    flow_net = {'source': [('a', 4), ('b',3)],
                'b': [('a', 2), ('sink', 2)],
                'a': [('sink', 3)],
                'sink':[]}
    run_network(flow_net)
    
def test1():
    """
    test to determine if spiking flow
    net algorithms works correct
    """
    # example flow net 
    flow_net = {'source': [('a', 2), ('b',4)],
                'b': [('a', 2), ('sink', 2)],
                'a': [('sink', 4)],
                'sink':[]}
    run_network(flow_net)

def test2():
    """
    test to determine if spiking flow
    net algorithms works correct
    """
    # example flow net 
    flow_net = {'source': [('a', 3), ('b',3)],
                'b': [('a', 1), ('c', 2)],
                'a': [('d', 4)],
                'c' : [('d',1), ('sink',1)],
                'd': [('sink',5)],
                'sink':[]}

    run_network(flow_net)
    


    
    
def test3():
    """
    test to determine if spiking flow
    net algorithms works correct
    """
    # example flow net 
    flow_net = {'source': [('a', 5), ('b',5)],
                'b': [('c', 4)],
                'a': [('d', 2), ('b', 1), ('c',4)],
                'c' : [('d',3), ('sink',5)],
                'd': [('sink',5)],
                'sink':[]}
    run_network(flow_net)
    
def test4():
    flow_net = {'source': [('xTgY', 3), ('jKKG', 2), ('r76m', 6), ('KTaf', 7)], 
                           'bJcK': [('sink', 9)],  
                           'xTgY': [('FTnI', 5)], 
                           '1F7s': [('sink', 10)], 
                           'jKKG': [('bJcK', 9), ('1F7s', 2)], 
                           'r76m': [('sink', 10)], 
                           'KTaf': [('1y9X', 4)], 
                           'FTnI': [('sink', 2)], 
                           'GOer': [('FTnI', 9)], 
                           '1y9X': [('FTnI', 2), ('bJcK', 10), ('GOer', 7)], 'sink': []}
    run_network(flow_net)
def test5():
    flow_net = {'source': [('a', 16), ('c', 13)],
                'a': [('b', 12)],
                'c': [('a', 4), ('d', 14)],
                'b': [('sink', 20)],
                'd':[('b', 7), ('sink', 4)],
                'sink':[]}
    run_network(flow_net)
    
def test6():
    flow_net = {'source': [('a', 8), ('b', 6)],
                'a': [('c', 3) ,('d', 3)],
                'b': [('c', 3), ('d', 6)],
                'c': [('sink', 6)],
                'd':[('sink', 8)],
                'sink':[]}
    run_network(flow_net)
    

"""
Uncomment any of these calls (or both)
""" 
experiment()
#test0()       




# TODO:
# find a way to divide flow over outgoing arcs
# find a way preserve legal flow across nodes 
# find a way to run the deterministic network without priority







