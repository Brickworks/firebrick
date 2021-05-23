from scipy import sparse
import pandas as pd
import numpy as np

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
    index_vals = range(0,len(index_keys))
    index_map = {k: v for k, v in zip(index_keys, index_vals)}
    num_nodes = len(index_map.keys())
    connection_pairs = connections[['node1', 'node2']].applymap(
            lambda x: index_map[x], na_action='ignore').values

    # flip rows and columns and plop the repeat same data
    rows = np.hstack((connection_pairs[:, 0], connection_pairs[:, 1]))
    cols = np.hstack((connection_pairs[:, 1], connection_pairs[:, 0]))
    if var:
        data = np.hstack((connections[var].values, connections[var].values))
    else:
        data = np.ones(rows.shape)

    sparse_matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(num_nodes, num_nodes))
    return sparse_matrix, index_map


def coupling_from_conductances(connections):
    conductance_spmatrix, index_map = to_sparse_matrix(connections, 'C')
    diagonals = np.diagflat(-np.sum(conductance_spmatrix, axis=1))
    K_spmatrix = conductance_spmatrix + diagonals
    return K_spmatrix, index_map


def delete_indices_symmetric(spmatrix, indices_to_drop):
    ''' delete rows and columns at indices from sparse matrix '''
    rows_to_keep = list(set(range(spmatrix.shape[0]))-set(indices_to_drop))
    spmatrix = spmatrix[rows_to_keep,:]
    cols_to_keep = list(set(range(spmatrix.shape[1]))-set(indices_to_drop))
    spmatrix = spmatrix[:, cols_to_keep]
    return spmatrix


def delete_indices_vector(vector, indices_to_drop):
    ''' delete rows and columns at indices from sparse matrix '''
    to_keep = list(set(range(vector.shape[0]))-set(indices_to_drop))
    return vector[to_keep]


def get_boundary_nodes(nodes):
    boundary_nodes = nodes[nodes['type'] == 'boundary']
    return boundary_nodes


def get_source_nodes(nodes):
    source_nodes = nodes[nodes['external_load'] > 0]
    return source_nodes


def conductance(resistance):
    try:
        conductance = 1/resistance
    except ValueError:
        conductance = 0
    return conductance


def display_df(df, override_name=None):
    name = override_name or df.name
    if name:
        print(f'{name}\n{df}')
    else:
        print(f'{df}')
