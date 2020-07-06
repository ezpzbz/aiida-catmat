"""
Example of running VaspRelaxWorkChain on 
BCC Li 2x2x2 supercell
"""
import os
import sys
import click

from ase.io import read

from aiida.common import AttributeDict, NotExistent
from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit

VaspMultiStageWorkChain = WorkflowFactory('bjm.vasp_multistage') #pylint: disable=invalid-name
StructureData = DataFactory('structure') #pylint: disable=invalid-name
KpointsData = DataFactory("array.kpoints") #pylint: disable=invalid-name


def example_multistage_workchain_li(vasp_code):
    """Prepares the calculation and submits it to daemon """
    resources = {
        'num_machines': 1, 
        'num_mpiprocs_per_machine': 1
        }
    options = {'resources': resources}
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])

    incar = {
        'NPAR': 1,
        'GGA': 'PS',
        'ISMEAR': 0,
        'PREC': 'Normal',
        'ICHARG': 0,
        'NSW': 50,
        'LDAU': False
    }

    potential_family = 'PBE.54'
    # potential_mapping = {'Li': 'Li_sv'}
    
    # parser_settings = {'add_energies': True,
    #                    'add_forces': True,
    #                    'add_stress': True,
    #                    'add_bands': True}

    # parser_settings = {'add_energies': True}

    thisdir = os.getcwd()
    strc_path = os.path.join(thisdir, 'files', 'Li_bcc_222.cif')

    builder = VaspMultiStageWorkChain.get_builder()
    builder.base.code = vasp_code
    builder.structure = StructureData(ase=read(strc_path))
    builder.parameters = orm.Dict(dict=incar)
    # builder.settings = orm.Dict(dict={
    #     'parser_settings':parser_settings
    # })

    builder.base.potential_family = orm.Str(potential_family)
    # builder.base.potential_mapping = orm.Dict(dict=potential_mapping)
    builder.base.kpoints = kpoints

    builder.base.verbose = orm.Bool(True)


    builder.base.options = orm.Dict(dict=options)

    builder.metadata.label = 'BCC_Li_111'
    builder.metadata.description = 'VaspMultiStageWorkChain'
    builder.base.clean_workdir = orm.Bool(False)
    submit(builder)
    print('submitted VaspRelaxWorkChain!')

@click.command('cli')
@click.argument('codelabel')
def cli(codelabel):
    """Click interface"""
    try:
        code = orm.Code.get_from_string(codelabel)
    except NotExistent:
        print(f"<{codelabel}> does not exist!")
        sys.exit(1)
    example_multistage_workchain_li(code)

if __name__ == '__main__':
    cli() #pylint: disable=no-value-for-parameter