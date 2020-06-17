import numpy as np
from aiida.common.extendeddicts import AttributeDict
from aiida.manage.configuration import load_profile
from aiida.orm import Bool, Str, Code, Int, Float
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit
load_profile()


def launch_aiida(structure, code_string, resources,
                 label="SiC VASP calculation"):
    Dict = DataFactory('dict')
    KpointsData = DataFactory("array.kpoints")

    incar_dict = {
        'PREC': 'Accurate',
        'EDIFF': 1e-8,
        'IBRION': -1,
        'NELMIN': 5,
        'NELM': 100,
        'ENCUT': 240,
        'IALGO': 38,
        'ISMEAR': 0,
        'SIGMA': 0.01,
        'GGA': 'PS',
        'LREAL': False,
        'LCHARG': False,
        'LWAVE': False,
    }

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2], offset=[0, 0, 0])

    options = {'resources': resources}
            #    'account': '',
            #    'max_memory_kb': 1024000,
            #    'max_wallclock_seconds': 3600 * 10}

    potential_family = 'PBE.54'
    potential_mapping = {'Si': 'Si'}

    parser_settings = {'add_energies': True,
                       'add_forces': True,
                       'add_stress': True}

    code = Code.get_from_string(code_string)
    Workflow = WorkflowFactory('vasp.vasp')
    builder = Workflow.get_builder()
    builder.code = code
    builder.parameters = Dict(dict=incar_dict)
    builder.structure = structure
    builder.settings = Dict(dict={'parser_settings': parser_settings})
    builder.potential_family = Str(potential_family)
    builder.potential_mapping = Dict(dict=potential_mapping)
    builder.kpoints = kpoints
    builder.options = Dict(dict=options)
    builder.metadata.label = label
    builder.metadata.description = label
    builder.clean_workdir = Bool(False)

    node = submit(builder)
    return node

def get_structure():
    """
    Set up Si primitive cell

    fcc Si:
       3.9
       0.5000000000000000    0.5000000000000000    0.0000000000000000
       0.0000000000000000    0.5000000000000000    0.5000000000000000
       0.5000000000000000    0.0000000000000000    0.5000000000000000
    Si
       1
    Cartesian
    0.0000000000000000  0.0000000000000000  0.0000000000000000

    """

    structure_data = DataFactory('structure')
    alat = 3.9
    lattice = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]]) * alat
    structure = structure_data(cell=lattice)
    positions = [[0.0, 0.0, 0.0]]
    for pos_direct in positions:
        pos_cartesian = np.dot(pos_direct, lattice)
        structure.append_atom(position=pos_cartesian, symbols='Si')
    return structure

# def get_structure_SiC():
#     """Set up SiC cell

#     Si C
#        1.0
#          3.0920072935808083    0.0000000000000000    0.0000000000000000
#         -1.5460036467904041    2.6777568649277486    0.0000000000000000
#          0.0000000000000000    0.0000000000000000    5.0733470000000001
#      Si C
#        2   2
#     Direct
#        0.3333333333333333  0.6666666666666665  0.4995889999999998
#        0.6666666666666667  0.3333333333333333  0.9995889999999998
#        0.3333333333333333  0.6666666666666665  0.8754109999999998
#        0.6666666666666667  0.3333333333333333  0.3754109999999997

#     """

#     StructureData = DataFactory('structure')
#     a = 3.092
#     c = 5.073
#     lattice = [[a, 0, 0],
#                [-a / 2, a / 2 * np.sqrt(3), 0],
#                [0, 0, c]]
#     structure = StructureData(cell=lattice)
#     for pos_direct, symbol in zip(
#             ([1. / 3, 2. / 3, 0],
#              [2. / 3, 1. / 3, 0.5],
#              [1. / 3, 2. / 3, 0.375822],
#              [2. / 3, 1. / 3, 0.875822]), ('Si', 'Si', 'C', 'C')):
#         pos_cartesian = np.dot(pos_direct, lattice)
#         structure.append_atom(position=pos_cartesian, symbols=symbol)
#     return structure


def main(code_string, resources):
    structure = get_structure()
    launch_aiida(structure, code_string, resources)


if __name__ == '__main__':
    code_string = 'vasp5std@ovan'
    resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 2}
    main(code_string, resources)