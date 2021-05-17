import numpy as np
import pandas as pd

from utils import(
    pairs2square,
    get_boundary_nodes,
)

def conductance_matrix(nodes, connections):
    sq_mat = pairs2square(connections, 'C')
    sum = sq_mat.sum()
    i = np.identity(len(sum.values))
    eig = i * sum.values  # sums along diagonal
    print(sq_mat)
    c = -sq_mat+eig
    c.name = 'C'

    # drop rows and columns of boundary conditions
    boundary_nodes = get_boundary_nodes(nodes)['name'].to_list()
    c = c[~c.index.isin(boundary_nodes)]
    c = c.drop(boundary_nodes, axis=1)
    return c


def heat_source_sink_matrix(nodes, connections):
    c = pairs2square(connections, 'C')
    q_sources = nodes['dissipated_heat'].fillna(0)
    q_sources.index = nodes['name'].values
    q_sources = q_sources.reindex(c.index.values)
    q_sources.name = 'Q'
    for _, bc in get_boundary_nodes(nodes).iterrows():
        T0 = bc['T0']
        q_bc = c[bc['name']] * T0
        q_sources = q_sources.add(q_bc.values)
        print(c[bc['name']])
    # drop rows and columns of boundary conditions
    boundary_nodes = get_boundary_nodes(nodes)['name'].to_list()
    q_sources = q_sources[~q_sources.index.isin(boundary_nodes)]
    return q_sources


def temperatures(conductance_matrix, source_sink_matrix):
    c_inv = np.linalg.inv(conductance_matrix.values)
    q = source_sink_matrix.values
    temps = np.dot(c_inv, q)
    return pd.Series(temps, index=conductance_matrix.index, name='T_C')


def steady_state(nodes, connections):
    print('-- STEADY STATE ANALYSIS --')
    print('resistance matrix:')
    print(pairs2square(connections, 'R', fillna='inf'))
    c_mat = conductance_matrix(nodes, connections)
    q_mat = heat_source_sink_matrix(nodes, connections)
    print('Conductance matrix, C:')
    print(c_mat)
    print('Source & BC matrix, Q:')
    print(q_mat)
    temps = temperatures(c_mat, q_mat)
    print('Temperatures (C)')
    print(temps)
