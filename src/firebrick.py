import pandas as pd
import numpy as np


def conductance(resistance):
    try:
        conductance = 1/resistance
    except ValueError:
        conductance = 0
    return conductance


def pairs2square(connections, var, fillna=0):
    pairs = connections[['node1', 'node2', var]]
    s = pairs.pivot(*pairs)
    sq_mat = s.add(s.T, fill_value=0).fillna(fillna)
    return sq_mat


def conductance_matrix(connections):
    sq_mat = pairs2square(connections, 'C')
    sum = sq_mat.sum()
    i = np.identity(len(sum.values))
    eig = i * sum.values  # sums along diagonal
    c = -sq_mat+eig
    c.name = 'C'

    # drop rows and columns of boundary conditions
    boundary_nodes = get_boundary_nodes(nodes)['name'].to_list()
    c = c[~c.index.isin(boundary_nodes)]
    c = c.drop(boundary_nodes, axis=1)
    return c


def heat_source_sink_matrix(nodes, connections):
    c = pairs2square(connections, 'C')
    q_sources = nodes['dissipated_heat_w'].fillna(0)
    q_sources.index = nodes['name'].values
    q_sources = q_sources.reindex(c.index.values)
    q_sources.name = 'Q'
    for _, bc in get_boundary_nodes(nodes).iterrows():
        T0 = bc['T0_C']
        q_bc = c[bc['name']] * T0
        q_sources = q_sources.add(q_bc.values)
    # drop rows and columns of boundary conditions
    boundary_nodes = get_boundary_nodes(nodes)['name'].to_list()
    q_sources = q_sources[~q_sources.index.isin(boundary_nodes)]
    return q_sources


def get_boundary_nodes(nodes):
    boundary_nodes = nodes[nodes['type'] == 'boundary']
    return boundary_nodes


def get_source_nodes(nodes):
    source_nodes = nodes[nodes['dissipated_heat_w'] > 0]
    return source_nodes


def temperatures(conductance_matrix, source_sink_matrix):
    c_inv = np.linalg.inv(conductance_matrix.values)
    q = source_sink_matrix.values
    temps = np.dot(c_inv, q)
    return pd.Series(temps, index=conductance_matrix.index, name='T_C')


def main():
    nodes = pd.read_csv('nodes.csv')
    connections = pd.read_csv('connections.csv')
    connections['C'] = connections['R'].apply(conductance)
    print(nodes)
    print(connections)
    print('boundary nodes: %s' % get_boundary_nodes(nodes)['name'].to_list())
    print('source nodes: %s' % get_source_nodes(nodes)['name'].to_list())
    print('resistance matrix:')
    print(pairs2square(connections, 'R', fillna='inf'))
    c_mat = conductance_matrix(connections)
    q_mat = heat_source_sink_matrix(nodes, connections)
    print('Conductance matrix, C:')
    print(c_mat)
    print('Source & BC matrix, Q:')
    print(q_mat)
    temps = temperatures(c_mat, q_mat)
    print('Temperatures (C)')
    print(temps)


if __name__ == '__main__':
    main()