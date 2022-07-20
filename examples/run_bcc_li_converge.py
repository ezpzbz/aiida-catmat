"""Example of running VaspRelaxWorkChain on
BCC Li 2x2x2 supercell
"""
import os
import sys
import click

from ase.io import read

from aiida.common import NotExistent
from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit

VaspConvergeWorkChain = WorkflowFactory('catmat.vasp_converge')  #pylint: disable=invalid-name
StructureData = DataFactory('structure')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name


def example_multistage_workchain_li(vasp_code):
    """Prepares the calculation and submits it to daemon """

    incar = {
        'NPAR': 1,
        'GGA': 'PS',
        'ISPIN': 1,
        'LDAU': False,
    }

    potential_family = 'PBE.54'

    thisdir = os.getcwd()
    strc_path = os.path.join(thisdir, 'files', 'Li_bcc_222.cif')

    builder = VaspConvergeWorkChain.get_builder()
    builder.vasp_base.vasp.code = vasp_code
    structure = StructureData(ase=read(strc_path))
    builder.structure = structure
    builder.parameters = orm.Dict(dict=incar)
    builder.protocol_tag = orm.Str('S_test')
    builder.potential_family = orm.Str(potential_family)

    builder.encut_list = orm.List(list=[200, 300, 400, 500])
    kspacing_list = [0.2, 0.3, 0.5]
    builder.kspacing_list = orm.List(list=kspacing_list)
    builder.threshold = orm.Float(3.0)
    offset = orm.List(list=[0, 0, 0])
    builder.offset = offset

    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(kspacing_list[-1], offset=offset.get_list())

    builder.vasp_base.vasp.kpoints = kpoints
    builder.vasp_base.vasp.metadata.options.resources = {
        'num_machines': 1,
        'num_cores_per_machine': 1,
        'num_mpiprocs_per_machine': 1,
    }

    builder.metadata.label = 'BCC_Li_111'
    builder.metadata.description = 'VaspMultiStageWorkChain'
    submit(builder)
    print('submitted VaspRelaxWorkChain!')


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
