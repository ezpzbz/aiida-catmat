"""
Example of running VaspRelaxWorkChain on
BCC Li 2x2x2 supercell
"""
import sys
import click

from aiida.common import NotExistent
from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit

VaspMultiStageWorkChain = WorkflowFactory('bjm.vasp_multistage')  #pylint: disable=invalid-name
StructureData = DataFactory('structure')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name


def example_multistage_workchain_li(vasp_code):
    """Prepares the calculation and submits it to daemon """

    incar = {
        'NCORE': 40,
        'ENCUT': 500,
        'ALGO': 'Normal',
        'GGA': 'PS',
        'ADDGRID': True,
        'LDAU': True,
    }

    # structure = orm.load_node(3477)
    structure = orm.load_node(15168)  #222 supercell

    builder = VaspMultiStageWorkChain.get_builder()
    builder.vasp_base.vasp.code = vasp_code
    builder.structure = structure
    builder.parameters = orm.Dict(dict=incar)
    builder.protocol_tag = orm.Str('S_conv')
    builder.potential_family = orm.Str('PBE.54')

    builder.kspacing = orm.Float(0.4)
    builder.kgamma = orm.Bool(True)

    builder.vasp_base.vasp.metadata.options.resources = {'parallel_env': 'mpi', 'tot_num_mpiprocs': 40}
    builder.vasp_base.vasp.metadata.options.withmpi = True
    builder.vasp_base.vasp.metadata.options.max_wallclock_seconds = 4 * 60 * 60
    builder.vasp_base.vasp.metadata.options.max_memory_kb = 50331648
    builder.vasp_base.vasp.metadata.options.account = 'Faraday_CATMAT'
    builder.vasp_base.vasp.metadata.options.queue_name = 'Gold'
    builder.vasp_base.vasp.metadata.options.custom_scheduler_commands = '#$ -ac allow=A'

    builder.metadata.label = 'Li2FeP2S6_222'
    builder.metadata.description = 'TEST_MICHAEL_A_NODE'
    submit(builder)
    print('submitted VaspMultiStageWorkChain!')


@click.command('cli')
@click.argument('codelabel')
def cli(codelabel):
    """Click interface"""
    try:
        code = orm.Code.get_from_string(codelabel)
    except NotExistent:
        print(f'<{codelabel}> does not exist!')
        sys.exit(1)
    example_multistage_workchain_li(code)


if __name__ == '__main__':
    cli()  #pylint: disable=no-value-for-parameter
