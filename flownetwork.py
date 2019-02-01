import matplotlib.pyplot as plt
import random, string, itertools
from collections import deque
plt.style.use('ggplot')



def Edmonds_Karp(gen_flow_net):
    """
    Calculates reference max flow in network
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
        
    while augmenting_path:
        q = deque([])
        q.append('source')
        pred = dict() # store each edge taken
        while len(q) != 0:
            curr = q.popleft()
            for neighbour, capacity, flow, rev_flow in flow_net[curr]:
                if neighbour not in pred.keys() and capacity > flow:
                    pred[neighbour] = (curr, neighbour, capacity, flow, rev_flow)
                    q.append(neighbour)
            
        if 'sink' in pred.keys(): 
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
               
    return max_flow

def label_flow_net(flow_net):
    """
    Function that labels connections
    such that nodes closer to the
    sink have a lower value
    """
    new_flow_net = dict()
    q = deque([('source', 0)])
    p_list, visited = dict(), []
    max_l = -float('inf')
    while len(q) != 0:
        node, l = q.popleft()
        if l > max_l:
            max_l = l
        if node not in visited:
            visited.append(node)
            p_list[node] = l
            neighbours = flow_net[node]
            [q.append((n,l+1)) for n, cap in neighbours]
     # reverse l values
    rev_list = list(range(0,max_l + 1))
    
    rev_list.reverse()
    for node, value in p_list.items():
        p_list[node] = rev_list[value]
    p_list['sink'] = 0
    for node, l in p_list.items():
        neighbours = flow_net[node]
        new_flow_net[node] = [(n, cap, 1 + p_list[n]) for n, cap in neighbours]  
    return new_flow_net
    
def cycle_exists(G):
    """
    Function that checks if a cycle exists 
    in a given graph G
    """
    depth_limit = 0                     # - G is a directed graph
    color = { u : "white" for u in G  }  # - All nodes are initially white
    found_cycle = [False]                # - Define found_cycle as a list so we can change
                                         # its value per reference, see:
                                         # http://stackoverflow.com/questions/11222440/python-variable-reference-assignment
    for u in G:                          # - Visit all nodes.
        if color[u] == "white":
            dfs_visit(G, u, color, found_cycle, depth_limit)
        if found_cycle[0]:
            break
    return found_cycle[0]
 
#-------
 
def dfs_visit(G, u, color, found_cycle, depth_limit):
    """
    Function that runs a depth-limited DFS search 
    """
    if depth_limit == 10:
        found_cycle[0] = False
        return
    if found_cycle[0]:                          # - Stop dfs if cycle is found.
        return
    color[u] = "gray"                           # - Gray nodes are in the current path
    for v in G[u]:                              # - Check neighbors, where G[u] is the adjacency list of u.
        if color[v] == "gray":                  # - Case where a loop in the current path is present.  
            found_cycle[0] = True       
            return
        if color[v] == "white":                 # - Call dfs_visit recursively.   
            dfs_visit(G, v, color, found_cycle, depth_limit+1)
    color[u] = "black"      

def is_connected(G, node, visited, depth_limit=0):
    """
    Function that checks whether a graph is connected,
    by performing a single depth-limited DFS
    """
 
    if depth_limit == 20:
        return False
    if all(list(visited.values())):
        return True
    visited[node] = True
    connected = []
    for child in G[node]:
        connected += [is_connected(G, child, visited, depth_limit+1)]
    return all(connected)

def generate_nodes(nr_nodes):
    """
    function that generates a list of nodes of size
    "nr_nodes" where the intermediate nodes have an identifier length
    of 4
    """
    return [''.join(random.choice(string.ascii_letters + string.digits) for n in range(4)) \
         for i in range(nr_nodes)] 
    
    
def generate_flow_net(n, m, max_capacity):
    """
    Function that generates a flow network with a certain max capacity
     has the following structure: {'node': [(neighbour, c),.., (neighbour,c)]}
    A dictionary , where the keys are strings that denote the name of the node.
    The source and sink nodes will have the names 'source' and 'sink'. The 
    intermediate nodes will have a unique identifier which is a randomly generated
    string. 
    """

    # Generate the nodes
    nodes = generate_nodes(n)
   
    # Generate all possible connections
    connections = [(i,j) for i, j in itertools.product(nodes,nodes)]
    # remove self loops
    connections = [(i,j) for i, j in connections if i != j]

 
    # create graph 
    """
    First we shuffle the connections and pick the first nr_edges
    connections as candidate edges in the network. The network
    is defined according to 5 criteria:
    1: Is there at least one outgoing edge from source to intermediate node?
    2: Is there at least one incoming edge from intermediate node to sink?
    3: Does each intermediate node have atleast one incoming and one outgoing edge?
    4: Does the constructed network have no cycles
    5: Is the constructed network a fully connected component?
    If these criteria are met, we have a valid connectivity pattern for our
    flow network and we terminate the while loop.
    """
    flow_graph, finished, cur_conn = None, False, connections

    while not finished:
        flow_graph = dict()
        cur_conn = connections
        random.shuffle(cur_conn) # randomly permute connections
        cur_conn = cur_conn[:m] # pick first nr_edges connections
        # connect source to all nodes with  no incoming edges
        source_neighbours = []
        for node in nodes:
            if len([j for i, j in cur_conn if j == node]) == 0:
                source_neighbours.append(node)
        if len(source_neighbours) == 0: # pick arbitrairy number of vertices
            rand_indcs = random.sample(range(len(nodes)), random.randint(1,n))
            source_neighbours = [nodes[i] for i in sorted(rand_indcs)]
            
        cur_conn += [('source', j) for j in source_neighbours]
        sink_neighbours = []
        for node in nodes:
            if len([i for i, j in cur_conn if i == node]) == 0:
                sink_neighbours.append(node)
        if len(sink_neighbours) == 0: # pick arbitrairy number of vertices
            rand_indcs = random.sample(range(len(nodes)), random.randint(1,n))
            sink_neighbours = [nodes[i] for i in sorted(rand_indcs)]
            
        cur_conn += [(i, 'sink') for i in sink_neighbours]
        
        # construct graph
        for node in ['source'] + nodes + ['sink']:
           flow_graph[node] = [j for i, j in cur_conn if i == node]
       
        # check if there are cycles and 
        if cycle_exists(flow_graph):
            continue
        
        # check if graph is connected
        visited = dict([(node, False) for node in nodes])
        if not is_connected(flow_graph, 'source', visited):
            continue
        finished = True
     
        
    # create flow net with capacities
    flow_net = dict()
    for node in ['source'] + nodes + ['sink']:
        flow_net[node] = [(i, random.randint(2,max_capacity)) for i in flow_graph[node]]
    max_flow = Edmonds_Karp(flow_net) # calculate reference max flow
    return (flow_net, max_flow)
        