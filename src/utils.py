def pairs2square(connections, var, fillna=0):
    pairs = connections[['node1', 'node2', var]]
    # pivot to translate pairs into a matrix
    pivot_table = pairs.pivot(*pairs)
    # add the transpose of the pivot to make the matrix square
    square_matrix = pivot_table.add(pivot_table.T, fill_value=0)
    # fill nans
    square_matrix = square_matrix.fillna(fillna)
    return square_matrix


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
