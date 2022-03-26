import time
import logging
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import ode
from scipy.optimize import fsolve

''' ASSUMPTIONS
1. convection heat transfer is linear
    This is reasonable for airflows with velocity under 5 m/s
2. boundary nodes do not change temperature
3. thermal resistances are constant between nodes
4. assume linear relationships for steady state analysis
'''

log = logging.getLogger()


def pairs2square(connections, var, fillna=0):
    ''' turn a sparse coordinate matrix (row, col, data) into a square matrix
    using a pandas backend. only supports numeric data
    '''
    pairs = connections[['node1', 'node2', var]]
    # pivot to translate pairs into a matrix
    pivot_table = pairs.pivot(*pairs)
    # add the transpose of the pivot to make the matrix square
    square_matrix = pivot_table.add(pivot_table.T, fill_value=0)
    # fill nans
    square_matrix = square_matrix.fillna(fillna)
    return square_matrix


def to_sparse_matrix(connections, var=None):
    ''' turn a connections dataframe into a square sparse matrix in
        coordinate format [(row,col) data] using scipy backend
    'node1' --> row coordinates
    'node2' --> col coordinates
      var   --> data at coordinates
    '''
    index_keys = pd.unique(connections[['node1', 'node2']].values.ravel('K'))
    index_vals = range(0, len(index_keys))
    index_map = {k: v for k, v in zip(index_keys, index_vals)}
    num_nodes = len(index_map.keys())
    connection_pairs = connections[['node1', 'node2'
                                    ]].applymap(lambda x: index_map[x],
                                                na_action='ignore').values

    # flip rows and columns and plop the repeat same data
    rows = np.hstack((connection_pairs[:, 0], connection_pairs[:, 1]))
    cols = np.hstack((connection_pairs[:, 1], connection_pairs[:, 0]))
    if var:
        data = np.hstack((connections[var].values, connections[var].values))
    else:
        data = np.ones(rows.shape)

    sparse_matrix = sparse.coo_matrix((data, (rows, cols)),
                                      shape=(num_nodes, num_nodes))
    return sparse_matrix, index_map


def coupling_from_conductances(connections):
    conductance_spmatrix, index_map = to_sparse_matrix(connections, 'C')
    diagonals = np.diagflat(-np.sum(conductance_spmatrix, axis=1))
    K_spmatrix = conductance_spmatrix + diagonals
    return K_spmatrix, index_map


def delete_indices_symmetric(spmatrix, indices_to_drop):
    ''' delete rows and columns at indices from sparse matrix '''
    rows_to_keep = list(set(range(spmatrix.shape[0])) - set(indices_to_drop))
    spmatrix = spmatrix[rows_to_keep, :]
    cols_to_keep = list(set(range(spmatrix.shape[1])) - set(indices_to_drop))
    spmatrix = spmatrix[:, cols_to_keep]
    return spmatrix


def delete_indices_vector(vector, indices_to_drop):
    ''' delete rows and columns at indices from sparse matrix '''
    to_keep = list(set(range(vector.shape[0])) - set(indices_to_drop))
    return vector[to_keep]


def get_boundary_nodes(nodes):
    boundary_nodes = nodes[nodes['type'] == 'boundary']
    return boundary_nodes


def get_source_nodes(nodes):
    source_nodes = nodes[nodes['external_load'] > 0]
    return source_nodes


def conductance(resistance):
    try:
        conductance = 1 / resistance
    except ValueError:
        conductance = 0
    return conductance


def display_df(df, override_name=None):
    name = override_name or df.name
    if name:
        print(f'{name}\n{df}')
    else:
        print(f'{df}')


class Node():
    ''' points that have temperature and transfer heat '''
    def __init__(self, node):
        self.name = node['name']
        self.description = node['comment']
        self.type = node['type']
        self.T = node['T0']
        self.T0 = node['T0']
        self.connections = None
        if self.type == 'diffusion':
            self.thermal_mass = node['thermal_mass']
        else:
            self.thermal_mass = 0
        self.external_load = node['external_load'] if ~np.isnan(
            node['external_load']) else 0

    def __repr__(self):
        return self.name

    def __str__(self):
        return f'{self.name} ({self.type})'


class NodeMap():
    ''' a collection of connected nodes '''
    def __init__(self, nodes, connections):
        self.nodes = self._setup_nodes(nodes)
        self.conductances, self.connections, self.connection_types, self.index = \
            self._setup_connections(connections)
        self.loads = self._setup_external_loads(self.nodes)
        self.Q, self.K, self.Kmap = self._setup_boundary_conditions(
            self.nodes, self.conductances, self.loads, self.index)

    def __str__(self):
        return f'{self.conductances}'

    def _setup_nodes(self, nodes):
        node_list = []
        for _, node in nodes.iterrows():
            n = Node(node)
            node_list.append(n)
        return node_list

    def _setup_connections(self, connections):
        # adjacency matrix (which nodes touch)
        adj_spmatrix, index_map = to_sparse_matrix(connections)
        # connection classification matrix (type of connection)
        connections['typeenum'] = connections['type'].apply(
            self._connectiontype2enum)
        type_spmatrix, _ = to_sparse_matrix(connections, 'typeenum')
        # conductance coupling matrix (aka K)
        K_spmatrix, _ = coupling_from_conductances(connections)
        return K_spmatrix, adj_spmatrix, type_spmatrix, index_map

    def _connectiontype2enum(self, connection_type):
        if connection_type == 'conduction':
            return 1
        elif connection_type == 'convection':
            return 2
        elif connection_type == 'radiation':
            return 3
        else:
            return 0

    def _enum2connectiontype(self, enum):
        if enum == 1:
            return 'conduction'
        elif enum == 2:
            return 'convection'
        elif enum == 3:
            return 'radiation'
        else:
            return 'undefined'

    def _setup_external_loads(self, nodes):
        ''' just external loads
        in KT=Q format, Q is negative when heat is entering the node
        '''
        Q = np.zeros((len(nodes), 1))
        for index, node in enumerate(nodes):
            Q[index] -= node.external_load
        return Q

    def _setup_boundary_conditions(self, nodes, K, Q, index_map):
        ''' just boundary conditions
        in KT=Q format, Q is negative when heat is entering the node
        '''
        bc_indices = []
        bc_nodenames = []
        # add boundary condition loads
        for index, node in enumerate(nodes):
            if node.type == 'boundary':
                bc_index = index_map[node.name]
                bc_indices.append(bc_index)
                bc_nodenames.append(node.name)
                Q[index] += node.T * np.sum(K[bc_index, bc_index])
        # drop boundary nodes from K, Q, index
        K = delete_indices_symmetric(K, bc_indices)
        Q = delete_indices_vector(Q, bc_indices)
        for nodename in bc_nodenames:
            index_map.pop(nodename, None)
        # reindex the index map
        Kmap = {}
        for newindex, nodename in enumerate(index_map.keys()):
            Kmap[nodename] = newindex
        return Q, K, Kmap

    def reset_to_initial_temperatures(self):
        for node in self.nodes:
            node.T = node.T0

    def set_temperatures(self, new_temps_dict):
        ''' {'node name': new_temp} '''
        for node in self.nodes:
            node.T = new_temps_dict[node.name]

    def get_temperatures(self):
        temperatures = {}
        for node in self.nodes:
            temperatures[node.name] = node.T
        return temperatures

    def steady_state(self):
        ''' solve KT=Q for T '''
        Kinv = np.linalg.inv(self.K)
        temps = np.dot(Kinv, self.Q).A1
        new_temps_dict = {}
        for nodename, nodeindex in self.Kmap.items():
            new_temps_dict[nodename] = temps[nodeindex]
        return new_temps_dict

    def step(nodemap):
        ''' step thermal model forward in time '''
        # # integrate diffusion nodes
        # nodes = diffusion(nodes)
        # # relax arithmetic nodes
        # nodes = relaxation_error(nodes)
        return nodemap


# def diffusion(nodes):
#     ''' transient response of diffusion nodes '''
#     # transfer heat to/from surrounding nodes
#     return nodes


# def relaxation_error(nodes):
#     ''' solve for arithmetic node temperatures
#     arithmetic nodes are held fixed as boundary nodes while solving for
#     diffusion node temperatures, then relaxed to calculate their
#     equilibrium temperatures
#     '''
#     # sum of heat flows in/out of nodes due to energy exchange with
#     # neighboring nodes
#     return nodes


# def transient(nodes, connections, duration_s, dt_s, progress_pct=10):
#     # solver = ode(diffusion).set_integrator(
#     #     'dopri5', atol=1e-6, rtol=1e-3, nsteps=1000)
#     nodemap = NodeMap(nodes, connections)
#     progress = 0
#     timespan = pd.timedelta_range(
#         start='0 seconds',
#         end=f'{duration_s} seconds',
#         periods=np.ceil(duration_s / dt_s))
#     data = []
#     for t_index, time in enumerate(timespan):
#         if time > pd.Timedelta('0 seconds'):
#             # after first time step
#             if time % pd.Timedelta(f'{duration_s/progress_pct} seconds') < pd.Timedelta(f'{dt_s} seconds'):
#                 progress += progress_pct
#                 print(f'Progress {progress}% ({time})')
#             nodes = step(nodemap)
#         temperatures = nodemap.temperatures()
#         data.append(temperatures)
#     timeseries = pd.DataFrame(data, columns=nodes.index, index=timespan)
#     return timeseries


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
