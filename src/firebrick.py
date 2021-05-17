import pandas as pd

from utils import (
    conductance,
    get_boundary_nodes,
    get_source_nodes,
)
from steadystate import steady_state
from transient import transient

''' ASSUMPTIONS
1. convection heat transfer is linear
    This is reasonable for airflows with velocity under 5 m/s
2. boundary nodes do not change temperature
3. thermal resistances are constant between nodes
4. assume linear relationships for steady state analysis
'''


def main():
    nodes = pd.read_csv('nodes.csv')
    connections = pd.read_csv('connections.csv')
    connections['C'] = connections['R'].apply(conductance)
    print(nodes)
    print(connections)
    print('boundary nodes: %s' % get_boundary_nodes(nodes)['name'].to_list())
    print('source nodes: %s' % get_source_nodes(nodes)['name'].to_list())
    steady_state(nodes, connections)
    transient(nodes, connections, 1000, 0.1)


if __name__ == '__main__':
    main()
