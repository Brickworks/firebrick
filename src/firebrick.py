import pandas as pd

from utils import (
    conductance,
    get_boundary_nodes,
    get_source_nodes,
)
from nodetools import NodeMap

''' ASSUMPTIONS
1. convection heat transfer is linear
    This is reasonable for airflows with velocity under 5 m/s
2. boundary nodes do not change temperature
3. thermal resistances are constant between nodes
4. assume linear relationships for steady state analysis
'''


def main():
    import time
    nodes = pd.read_csv('nodes.csv')
    connections = pd.read_csv('connections.csv')
    connections['C'] = connections['R'].apply(conductance)
    print(nodes)
    print(connections)
    print('boundary nodes: %s' % get_boundary_nodes(nodes)['name'].to_list())
    print('source nodes: %s' % get_source_nodes(nodes)['name'].to_list())
    tic = time.perf_counter()
    nodemap = NodeMap(nodes, connections)
    print(f'Conductance coupling matrix, K\n{nodemap.K}')
    print(f'Heat load and boundary condition vector, Q\n{nodemap.Q}')
    temps = nodemap.steady_state()
    toc = time.perf_counter()
    print(pd.Series(temps, index=nodemap.Kmap.keys(), name='Steady State Temps'))
    print(f'runtime: {toc - tic:0.4f} seconds')

if __name__ == '__main__':
    main()
