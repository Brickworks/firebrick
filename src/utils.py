from scipy.sparse import coo_matrix
import pandas as pd


def pairs2square(connections, var, fillna=0):
    pairs = connections[['node1', 'node2', var]]
    # pivot to translate pairs into a matrix
    pivot_table = pairs.pivot(*pairs)
    # add the transpose of the pivot to make the matrix square
    square_matrix = pivot_table.add(pivot_table.T, fill_value=0)
    # fill nans
    square_matrix = square_matrix.fillna(fillna)
    return square_matrix


def to_sparse_matrix(connections, var):
    ''' turn a connections dataframe into a sparse matrix in coordinate format
    'node1' --> row coordinates
    'node2' --> col coordinates
      var   --> data at coordinates
    '''
    index_keys = pd.unique(connections[['node1', 'node2']].values.ravel('K'))
    index_vals = range(0,len(index_keys))
    index_map = {k: v for k, v in zip(index_keys, index_vals)}

    connection_pairs = connections[['node1', 'node2']].applymap(
            lambda x: index_map[x], na_action='ignore').values
    data = connections[var].values

    sparse_matrix = coo_matrix(
        (data, (connection_pairs[:, 0], connection_pairs[:, 1])))
    return sparse_matrix, index_map


def get_boundary_nodes(nodes):
    boundary_nodes = nodes[nodes['type'] == 'boundary']
    return boundary_nodes


def get_source_nodes(nodes):
    source_nodes = nodes[nodes['dissipated_heat'] > 0]
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
