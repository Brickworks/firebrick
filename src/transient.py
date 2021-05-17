import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.integrate import ode
from scipy.optimize import fsolve

from utils import(
    pairs2square,
)


def connectiontype2enum(connection_type):
    if connection_type == 'conduction':
        return 1
    elif connection_type == 'convection':
        return 2
    elif connection_type == 'radiation':
        return 3
    else:
        return 0


def enum2connectiontype(enum):
    if enum == 1:
        return 'conduction'
    elif enum == 2:
        return 'convection'
    elif enum == 3:
        return 'radiation'
    else:
        return 'undefined'


def map_connections(values_matrix, connection_map, ConnectionClass):
    connection_matrix = values_matrix[connection_map].applymap(
        lambda x: ConnectionClass(x), na_action='ignore')
    return connection_matrix


class Connection():
    def __init__(self, conductance):
        self.type = 'undefined'
        self.conductance = conductance
        if self.conductance > 0:
            self.resistance = 1/self.conductance
        else:
            self.resistance = 0

    def __repr__(self):
        return self.conductance

    def __str__(self):
        return f'{self.conductance} ({self.type})'

    def flow(self, T1, T2):
        return 0


class ConductiveConnection(Connection):
    def __init__(self, conductance):
        Connection.__init__(self, conductance)
        self.type = 'conduction'

    def flow(self, T_other_end):
        return self.conductance * T_other_end


class ConvectiveConnection(Connection):
    def __init__(self, conductance):
        Connection.__init__(self, conductance)
        self.type = 'convection'

    def flow(self, T_other_end):
        return self.conductance * T_other_end


class RadiativeConnection(Connection):
    def __init__(self, conductance):
        Connection.__init__(self, conductance)
        self.type = 'radiation'

    def flow(self, T_other_end):
        return self.conductance * T_other_end**4


class Node():
    def __init__(self, node):
        self.name = node['name']
        self.description = node['comment']
        self.type = node['type']
        self.temp = node['T0']
        self.T0 = node['T0']
        self.connections = None
        if self.type == 'diffusion':
            self.thermal_mass = node['thermal_mass']
        else:
            self.thermal_mass = 0
        self.dissipated_heat = node['dissipated_heat'] if ~np.isnan(
            node['dissipated_heat']) else 0

    def __repr__(self):
        return self.name

    def __str__(self):
        return f'{self.name} ({self.type})'

    def dT_dt(self, nodes):
        dT_dt = 0
        for index, connection in self.connections.iteritems():
            other_node = nodes[index]
            connection_dT_dt = 0
            if connection.type != 'undefined':
                connection_dT_dt = connection.flow(other_node.temp)
                print(f'{connection_dT_dt:<6} {self.name:>10} -[{connection.type:^10}]-> {other_node.name}')
            dT_dt += connection_dT_dt
        return dT_dt


def diffusion():
    ''' transient response of diffusion nodes '''
    # transfer heat to/from surrounding nodes
    dT_dt = 0
    # add heat from source loads

    # scale by thermal mass to calculate temperature rate of change
    pass


def do_something():
    # Divide thermal connections between conductive (linear), convective (linear)
    # and radiative (nonlinear), and divide based on whether they link
    # diffusion-diffusion, diffusion-arithmetic, diffusion-boundary,
    # arithmetic-arithmetic, or arithmetic-boundary nodes
    pass


def relaxation_error():
    ''' solve for arithmetic node temperatures
    arithmetic nodes are held fixed as boundary nodes while solving for
    diffusion node temperatures, then relaxed to calculate their
    equilibrium temperatures
    '''
    # sum of heat flows in/out of nodes due to energy exchange with
    # neighboring nodes

    # add heat from source loads
    pass


def step():
    ''' step thermal model forward in time '''
    # integrate diffusion nodes

    # relax arithmetic nodes

    # update temperatures
    pass


def setup_nodes(nodes):
    node_list = []
    for index, node in nodes.iterrows():
        n = Node(node)
        node_list.append(n)
    return pd.Series(node_list, index=nodes['name'])


def setup_connections(connections):
    connections['typeenum'] = connections['type'].apply(connectiontype2enum)
    connection_map = pairs2square(
        connections, 'typeenum', fillna=connectiontype2enum('undefined'))
    # create masks for each connection type
    conduction_map = connection_map == connectiontype2enum('conduction')
    convection_map = connection_map == connectiontype2enum('convection')
    radiation_map = connection_map == connectiontype2enum('radiation')
    # populate connection map with connection objects
    conductance_matrix = pairs2square(connections, 'C')
    connection_matrix = conductance_matrix * np.nan
    connection_matrix = connection_matrix.fillna(map_connections(
        conductance_matrix, conduction_map, ConductiveConnection))
    connection_matrix = connection_matrix.fillna(map_connections(
        conductance_matrix, convection_map, ConvectiveConnection))
    connection_matrix = connection_matrix.fillna(map_connections(
        conductance_matrix, radiation_map, RadiativeConnection))
    connection_matrix = connection_matrix.fillna(Connection(0))
    return connection_matrix


def setup(nodes, connections):
    connection_matrix = setup_connections(connections)
    nodes = setup_nodes(nodes)
    # update nodes with connections
    for index, connection_table in connection_matrix.iteritems():
        nodes[index].connections = connection_table
    return nodes, connection_matrix


def transient(nodes, connections, duration_s, dt_s):
    print('-- TRANSIENT ANALYSIS --')
    solver = ode(diffusion).set_integrator(
        'dopri5', atol=1e-6, rtol=1e-3, nsteps=1000)
    nodes, connection_matrix = setup(nodes, connections)
    print('Nodes \n%s' % nodes)
    print('Connection Matrix \n%s' % connection_matrix)
    timeseries = pd.timedelta_range(
        start='0 seconds',
        end=f'{duration_s} seconds',
        periods=np.ceil(duration_s / dt_s))
    temperatures = pd.DataFrame(columns=nodes.keys(), index=timeseries)
    for name, node in nodes.iteritems():
        node.dT_dt(nodes)


if __name__ == '__main__':
    from utils import conductance
    nodes = pd.read_csv('nodes.csv')
    connections = pd.read_csv('connections.csv')
    connections['C'] = connections['R'].apply(conductance)
    transient(nodes, connections, 1000, 0.1)
