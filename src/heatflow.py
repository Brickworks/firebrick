import time
import logging
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

log = logging.getLogger()


def solve(nodesfile, connectionsfile):
    nodes = pd.read_csv(nodesfile)
    connections = pd.read_csv(connectionsfile)
    connections['C'] = connections['R'].apply(conductance)
    log.debug(nodes)
    log.debug(connections)
    log.debug('boundary nodes: %s' % get_boundary_nodes(nodes)['name'].to_list())
    log.debug('source nodes: %s' % get_source_nodes(nodes)['name'].to_list())
    tic = time.perf_counter()
    nodemap = NodeMap(nodes, connections)
    log.debug(f'Conductance coupling matrix, K\n{nodemap.K}')
    log.debug(f'Heat load and boundary condition vector, Q\n{nodemap.Q}')
    temps = nodemap.steady_state()
    toc = time.perf_counter()
    log.info(f'Steady State Temperatures\n{pd.Series(temps)}')
    log.debug(f'runtime: {toc - tic:0.4f} seconds')


if __name__ == '__main__':
    main()
