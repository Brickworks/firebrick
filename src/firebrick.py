#!/usr/bin/env python3
import click
import logging

import heatflow
from mission_control.dashboard import run_server


@click.group()
@click.option('-v',
              '--verbose',
              is_flag=True,
              help='Show more log statements.')
def cli(verbose):
    FORMAT = '%(asctime)-15s %(levelname)+8s: %(message)s'
    logging.basicConfig(format=FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%Z")
    log = logging.getLogger()

    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)


@cli.command()
@click.pass_context
@click.argument('nodes', type=click.Path(exists=True))
@click.argument('connections', type=click.Path(exists=True))
def thermal_network(ctx, nodes, connections):
    ''' Calculate steady state heat flow.

    \b
    NODES       path to a CSV of definitions of nodes.
    CONNECTIONS path to a CSV of definitions of connections between nodes.
    '''
    heatflow.solve(nodes, connections)


@cli.command()
@click.pass_context
@click.option('-t',
              '--telemetry_database',
              type=click.Path(exists=True),
              envvar="TELEMETRY_DATABSE",
              help='Path to telemetry CSV.')
@click.option('-p',
              '--port',
              nargs=1,
              default=8050,
              help='Server port on localhost to use.')
def dashboard(ctx, telemetry_database, port):
    ''' Start a telemetry dashboard server on localhost.
    '''
    run_server(port=port, debug=True, threaded=True)


if __name__ == '__main__':
    cli()
