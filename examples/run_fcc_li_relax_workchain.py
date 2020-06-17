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

VaspRelaxWorkChain = WorkflowFactory('vasp.relax') #pylint: disable=invalid-name
StructureData = DataFactory('structure') #pylint: disable=invalid-name
KpointsData = DataFactory("array.kpoints") #pylint: disable=invalid-name


def example_relax_workchain_li(vasp_code):
    """Prepares the calculation and submits it to daemon """
    incar_dict = {
        'PREC': 'Normal',
        'EDIFF': 1e-4,
        'NELMIN': 3,
        'NELM': 100,
        'ENCUT': 520,
        'ALGO': 'Fast',
        'LREAL': 'AUTO',
        'ISMEAR': 0,
        'SIGMA': 0.01,
        'GGA': 'PS',
        'ISPIN': 2,
        'LREAL': False,
        'LCHARG': False,
        'LWAVE': False,
    }

    resources = {
        'num_machines': 1, 
        'num_mpiprocs_per_machine': 2
        }
    options = {'resources': resources}
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])

    potential_family = 'PBE.54'
    potential_mapping = {'Li': 'Li_sv'}
    
    parser_settings = {'add_energies': True,
                       'add_forces': True,
                       'add_stress': True}

    thisdir = os.getcwd()
    strc_path = os.path.join(thisdir, 'files', 'Li_bcc_222.cif')
    

    # Relaxation settings
    relax = AttributeDict()
    relax.perform = orm.Bool(True)        # Turn on relaxation of the structure
    relax.force_cutoff = orm.Float(1e-4)  # Relax force cutoff
    relax.steps = orm.Int(10)             # Relax number of ionic steps
    relax.positions = orm.Bool(True)      # Relax atomic positions
    relax.shape = orm.Bool(True)          # Relax cell shape (alpha, beta, gamma)
    relax.volume = orm.Bool(True)         # Relax volume


    builder = VaspRelaxWorkChain.get_builder()
    builder.code = vasp_code
    builder.structure = StructureData(ase=read(strc_path))
    builder.parameters = orm.Dict(dict=incar_dict)
    builder.settings = orm.Dict(dict={
        'parser_settings':parser_settings
    })

    builder.potential_family = orm.Str(potential_family)
    builder.potential_mapping = orm.Dict(dict=potential_mapping)
    builder.kpoints = kpoints

    builder.relax = relax
    builder.verbose = orm.Bool(True)


    builder.options = orm.Dict(dict=options)

    builder.metadata.label = 'BCC_Li_111'
    builder.metadata.description = 'VaspRelaxWorkChain'
    builder.clean_workdir = orm.Bool(False)
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
    example_relax_workchain_li(code)

if __name__ == '__main__':
    cli() #pylint: disable=no-value-for-parameter