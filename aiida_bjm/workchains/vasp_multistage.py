"""
VaspMultiStageWorkChain: A general purpose and modular AiiDA workchain
developed in Morgan Materials Modelling Group in University of Bath.
"""
import os

import yaml

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, ToContext, append_, while_
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_bjm.calcfunctions import dict_merge
from aiida_bjm.utils import prepare_process_inputs

VaspBaseWorkChain = WorkflowFactory('vasp.base')  #pylint: disable=invalid-name
PotcarData = DataFactory('vasp.potcar')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name


def get_magmom(structure_pmg):
    """Construct MAGMOM tag from structure"""
    # Get default
    default_magmoms = []
    for specie in structure_pmg.species:
        if specie.Z > 56:
            default_magmoms.append(7)
        elif specie.Z > 20:
            default_magmoms.append(5)
        else:
            default_magmoms.append(0.6)

    # Get from structure
    strc_magmoms = []
    for site in structure_pmg:
        if hasattr(site.specie, 'spin'):
            strc_magmoms.append(site.specie.spin)
        else:
            strc_magmoms.append(0)
    # merge
    magmom = []
    for m1, m2 in zip(strc_magmoms, default_magmoms):  #pylint: disable=invalid-name
        if m1 != 0:
            magmom.append(m1 * m2)
        else:
            magmom.append(m2)
    magmom_dict = {}
    magmom_dict['MAGMOM'] = magmom
    return magmom_dict


# Get Hubbard parameters if DFT+U is requested
def get_hubbard(structure, hubbard_tag):
    """Constructs LDAU part of INCAR"""
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(thisdir, '..', 'data', 'hubbard_sets.yaml')
    with open(yaml_path, 'r') as file:
        hubbard_sets = yaml.safe_load(file)
    hubbard_params = hubbard_sets[hubbard_tag]

    structure_pmg = structure.get_pymatgen_structure(add_spin=True)

    if any(element.Z > 56 for element in structure_pmg.composition):
        lmaxmix = 6
    elif any(element.Z > 20 for element in structure_pmg.composition):
        lmaxmix = 4

    hubbard_dict = {'LDAU': True, 'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LMAXMIX': lmaxmix}

    LDAUU = []  #pylint: disable=invalid-name
    LDAUJ = []  #pylint: disable=invalid-name
    LDAUL = []  #pylint: disable=invalid-name

    kinds = structure.get_kind_names()
    for kind in kinds:
        kind_no_digit = ''.join(i for i in kind if not i.isdigit())
        if kind_no_digit in hubbard_params['LDAUU']:
            LDAUU.append(hubbard_params['LDAUU'][kind_no_digit])
            LDAUJ.append(hubbard_params['LDAUJ'][kind_no_digit])
            LDAUL.append(hubbard_params['LDAUL'][kind_no_digit])
        else:
            LDAUU.append(0)
            LDAUJ.append(0)
            LDAUL.append(-1)
    hubbard_dict.update({'LDAUU': LDAUU, 'LDAUJ': LDAUJ, 'LDAUL': LDAUL})
    return hubbard_dict


def get_potcar_mapping(structure, potcar_set_tag):
    """Cosntruct potcar_mapping
    :param structure: the structure object
    :param potcar_set_tag: tag the tells which potcar set should be used.
    :return: AiiDA Dict object
    """
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(thisdir, '..', 'data', 'potcar_sets.yaml')

    with open(yaml_path) as default:
        potcar_dict = yaml.safe_load(default)
        sel_potcars = potcar_dict[potcar_set_tag.value]

    mapping = {}
    kinds = structure.get_kind_names()
    for kind in kinds:
        kind_no_digit = ''.join(i for i in kind if not i.isdigit())
        mapping[kind] = sel_potcars[kind_no_digit]
    return mapping


def should_sort_structure(structure):
    """Checks if structure needs to be sorted"""
    import functools  #pylint: disable=import-outside-toplevel
    structure_pmg = structure.get_pymatgen_structure(add_spin=True)
    structure_pmg_sorted = structure.get_pymatgen_structure(add_spin=True)
    structure_pmg_sorted.sort()
    species = structure_pmg.species
    species_sorted = structure_pmg_sorted.species
    spin = []
    for s in species:  #pylint: disable=invalid-name
        spin.append(getattr(s, 'spin', 0))
    spin_sorted = []
    for s in species_sorted:  #pylint: disable=invalid-name
        spin_sorted.append(getattr(s, 'spin', 0))
    stat = []
    if functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, spin, spin_sorted), True):
        stat.append(False)
    else:
        stat.append(True)
    return all(stat)


@calcfunction
def setup_protocols(protocol_tag, structure, user_incar_settings):
    """Read stages from provided protocol file, and
    constructs initial INCARs from Materials Project
    sets."""
    # structure_pmg = structure.get_pymatgen_structure(add_spin=True)

    # Get user-defined stages and alternative settings from yaml file
    thisdir = os.path.dirname(os.path.abspath(__file__))
    protocol_path = os.path.join(thisdir, 'protocols', 'vasp', protocol_tag.value + '.yaml')
    with open(protocol_path, 'r') as protocol:
        protocol = yaml.safe_load(protocol)

    # User-defined INCAR settings passed to workchain.
    user_incar_settings = user_incar_settings.get_dict()

    if user_incar_settings['LDAU']:
        for key in protocol.keys():
            protocol[key]['LDAU'] = True

    # magmom = get_magmom(structure_pmg)

    # Check for LREAL
    nions = 0
    comp = structure.get_composition()
    for nion in comp.values():
        nions += nion
    if nions <= 8:
        lreal = {'LREAL': False}
    else:
        lreal = {'LREAL': 'Auto'}

    # Update MAGMOM and LDAU section in all stages!
    for key in protocol.keys():
        dict_merge(protocol[key], lreal)
        # dict_merge(protocol[key], magmom)
        dict_merge(protocol[key], user_incar_settings)

    return orm.Dict(dict=protocol)


@calcfunction
def set_kpoints(structure, kspacing, kgamma=orm.Bool(False)):
    """Set kpoints mesh from kspacing"""
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    if kgamma:
        kpoints.set_kpoints_mesh_from_density(kspacing.value)
    else:
        kpoints.set_kpoints_mesh_from_density(kspacing.value, offset=[0.5, 0.5, 0.5])
    return kpoints


@calcfunction
def sort_structure(structure):
    """Apply structure sorting in case it is called"""
    structure_pmg = structure.get_pymatgen_structure(add_spin=True)
    structure_pmg.sort()
    return orm.StructureData(pymatgen_structure=structure_pmg)


@calcfunction
def get_stage_incar(protocol, structure, stage_tag, hubbard_tag=None, prev_incar=None, modifications=None):
    """get INCAR for next stage"""
    next_incar = protocol[stage_tag.value]
    structure_pmg = structure.get_pymatgen_structure(add_spin=True)
    magmom = get_magmom(structure_pmg)
    dict_merge(next_incar, magmom)
    if hubbard_tag:
        hubbard = get_hubbard(structure, hubbard_tag.value)
        dict_merge(next_incar, hubbard)
    if prev_incar:
        param_list = ['ALGO', 'ISMEAR', 'SIGMA', 'SYMPREC', 'AMIN', 'ISYM', 'KPAR', 'LREAL']
        prev_incar = prev_incar.get_dict()
        # Update next incar with params from previous INCAR
        for param in param_list:
            if param in prev_incar:
                next_incar[param] = prev_incar[param]
        if prev_incar['IBRION'] == -1:
            next_incar['LREAL'] = False
    if modifications:
        modifications = modifications.get_dict()
        for key, value in modifications.items():
            next_incar[key] = value
    return orm.Dict(dict=next_incar)


@calcfunction
def extract_wrap_results(**all_outputs):
    """Exctract and wrap results for whole workchain"""
    results_dict = {}
    for key, value in all_outputs.items():
        results_dict[key] = value.get_dict()
        if 'complete_site_magnetizations' in results_dict[key]:
            del results_dict[key]['complete_site_magnetizations']
    return orm.Dict(dict=results_dict)


def get_last_input(workchain):
    """Get last input of a successful calculation"""
    calcjobs = []
    descendants = workchain.called_descendants
    for desc in descendants:
        if desc.process_label == 'VaspCalculation':
            calcjobs.append(desc)
    pks = [cjob.pk for cjob in calcjobs]
    last_index = pks.index(max(pks))
    return calcjobs[last_index].inputs.parameters


# @calcfunction
# def update_prev_incar(incar, modifications):
#     """Merge two aiida Dict objects."""
#     incar = incar.get_dict()
#     modifications = modifications.get_dict()
#     dict_merge(incar, modifications)
#     return orm.Dict(dict=incar)


#pylint: disable=inconsistent-return-statements
class VaspMultiStageWorkChain(WorkChain):
    """Multi Stage Workchain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            VaspBaseWorkChain,
            include=[
                'clean_workdir', 'max_iterations', 'vasp.code', 'vasp.restart_folder', 'vasp.metadata', 'vasp.potential'
            ],
            namespace='vasp_base'
        )
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.input('parameters', valid_type=orm.Dict)
        spec.input('vasp_base.vasp.kpoints', valid_type=orm.KpointsData, required=False)
        spec.input('kspacing', valid_type=orm.Float, required=False, help='distance to generate kponits mesh')
        spec.input('kgamma', valid_type=orm.Bool, required=False, help='gamma centered kpoints in kspacing case')
        spec.input('magmom', valid_type=orm.List, required=False, help='List of MAGMOM')
        spec.input('potential_family', valid_type=orm.Str, required=True)
        spec.input('potential_mapping', valid_type=orm.Dict, required=False)
        spec.input('settings', valid_type=orm.Dict, required=False)
        spec.input('protocol_tag', valid_type=orm.Str, required=False, default=lambda: orm.Str('S0R3S'))
        spec.input('hubbard_tag', valid_type=orm.Str, required=False, default=lambda: orm.Str('MPSet'))
        spec.input('max_stage_iteration', valid_type=orm.Int, default=lambda: orm.Int(2), required=False)
        spec.input(
            'potcar_set',
            valid_type=orm.Str,
            default=lambda: orm.Str('VASP'),
            help='Select which potcar set should be used to construct mappin. VASP or MPRelaxSet'
        )
        spec.input('restart_folder', valid_type=orm.RemoteData, required=False)
        spec.input(
            'vasp_base.vasp.metadata.options.parser_name',
            valid_type=str,
            default='vasp_base_parser',
            non_db=True,
            help='Parser of the calculation: the default is cp2k_advanced_parser to get the necessary info'
        )

        spec.outline(
            cls.initialize,
            while_(cls.should_run_next_stage)(
                cls.run_stage,
                cls.inspect_stage,
            ),
            cls.results,
        )
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        spec.exit_code(420, 'ERROR_NO_CALLED_WORKCHAIN', message='no called workchain detected')
        spec.exit_code(800, 'ERROR_PROTOCOL_TAG', message='protocol has no stage_0 tag!')
        spec.exit_code(801, 'ERROR_UNSUPPORTED_CALC', message='An unsupported calculation is requested!')
        spec.exit_code(802, 'ERROR_UNRECOVERABLE_FAILURE', message='VaspBaseWorkChain could not handle the failure!')
        spec.exit_code(803, 'ERROR_KPOINTSDATA_NOT_PROVIDED', message='KpoinstData is not provided in any form.')
        spec.exit_code(804, 'ERROR_NON_CONVERGED_GEOMETRY', message='KpoinstData is not provided in any form.')
        spec.output('structure', valid_type=orm.StructureData, required=False)
        spec.output('output_parameters', valid_type=orm.Dict, required=True)
        spec.output('magmom', valid_type=orm.List, required=False, help='List of MAGMOM')
        spec.output_namespace('final_incar', valid_type=orm.Dict, required=False, dynamic=True)

    # pylint: disable=too-many-branches
    def initialize(self):
        """Initialize inputs and settings"""
        # Initiate inputs and structure
        self.ctx.vasp_base = AttributeDict(self.exposed_inputs(VaspBaseWorkChain, 'vasp_base'))
        self.ctx.current_structure = self.inputs.structure

        # Handle kpoints
        if 'kspacing' in self.inputs:
            if 'kgamma' in self.inputs:
                if self.inputs.kgamma:
                    kpoints = set_kpoints( #pylint: disable=unexpected-keyword-arg
                        self.ctx.current_structure, self.inputs.kspacing, self.inputs.kgamma,
                        metadata={
                            'label':'set_kpoints',
                            'description': 'calcfuntion to construct kpoints from kspacing',
                            'call_link_label':'run_set_kpoints'
                        })
            else:
                kpoints = set_kpoints( #pylint: disable=unexpected-keyword-arg
                    self.ctx.current_structure, self.inpts.kspacing,
                    metadata={
                            'label':'set_kpoints',
                            'description': 'calcfuntion to construct kpoints from kspacing',
                            'call_link_label':'run_set_kpoints'
                    })
            self.ctx.vasp_base.vasp.kpoints = kpoints
        elif 'kpoints' in self.inputs.vasp_base.vasp:
            self.ctx.vasp_base.vasp.kpoints = self.inputs.vasp_base.vasp.kpoints
        else:
            return self.exit_codes.ERROR_KPOINTSDATA_NOT_PROVIDED  #pylint: disable=no-member

        # MAGMOM
        if 'magmom' in self.inputs:
            self.ctx.magmom = self.inputs.magmom

        self.ctx.hubbard_tag = None
        if self.inputs.parameters['LDAU']:
            self.ctx.hubbard_tag = self.inputs.hubbard_tag

        self.ctx.protocol = setup_protocols( #pylint: disable=unexpected-keyword-arg
            self.inputs.protocol_tag,
            self.ctx.current_structure,
            self.inputs.parameters,
            metadata={
                'label':'setup_protocol',
                'description': 'calcfuntion to get and setup INCAR for all stages.',
                'call_link_label':'run_setup_protocol'})

        # Settings
        if 'settings' in self.inputs:
            settings = AttributeDict(self.inputs.settings.get_dict())
        else:
            settings = AttributeDict({'ADDITIONAL_RETRIEVE_LIST': ['INCAR', 'OSZICAR']})
            self.inputs.settings = settings
        self.ctx.vasp_base.vasp.settings = self.inputs.settings
        self.ctx.all_outputs = {}
        self.ctx.stage_iteration = 0
        self.ctx.prev_incar = None
        self.ctx.modifications = None

        # Restart folder
        # It is useful if user wants to restart later!
        if 'restart_folder' in self.inputs:
            self.ctx.restart_folder = self.inputs.restart_folder
        else:
            self.ctx.restart_folder = None

        self.ctx.stage_idx = 0
        if f'stage_{self.ctx.stage_idx}' in self.ctx.protocol.get_dict():
            self.ctx.should_run_next_stage = True
        else:
            return self.exit_codes.ERROR_PROTOCOL_TAG  #pylint: disable=no-member

        # Get the requested calculation types
        self.ctx.stage_calc_types = {}
        for stage_tag in list(self.ctx.protocol.keys()):
            if self.ctx.protocol[stage_tag]['IBRION'] in [-1, 1, 2, 3]:
                if self.ctx.protocol[stage_tag]['IBRION'] == -1:
                    self.ctx.stage_calc_types[stage_tag] = 'static'
                else:
                    self.ctx.stage_calc_types[stage_tag] = 'relaxation'
            else:
                return self.ctx.exit_codes.ERROR_UNSUPPORTED_CALC

    def should_run_next_stage(self):
        """True if there is another stage to run"""
        return self.ctx.should_run_next_stage

    def run_stage(self):
        """Prepares and submits static calculations as long as they are needed."""
        self.ctx.stage_tag = f'stage_{self.ctx.stage_idx}'

        # Check if structure needs to be sorted and do it.
        if should_sort_structure(self.ctx.current_structure):
            sorted_strucure = sort_structure(self.ctx.current_structure, metadata={ #pylint: disable=unexpected-keyword-arg
                'label':'sort_structure',
                'description': 'calcfuntion to sort structure',
                'call_link_label':'run_sort_structure'
            })
            self.ctx.current_structure = sorted_strucure
            self.ctx.restart_folder = None
            del self.ctx['vasp_base']['vasp']['restart_folder']

        self.ctx.vasp_base.vasp.structure = self.ctx.current_structure

        self.inputs.potential_mapping = get_potcar_mapping( #pylint: disable=unexpected-keyword-arg
            self.ctx.current_structure,
            self.inputs.potcar_set)

        self.ctx.vasp_base.vasp.potential = PotcarData.get_potcars_from_structure(
            structure=self.ctx.current_structure,
            family_name=self.inputs.potential_family.value,
            mapping=self.inputs.potential_mapping
        )

        # Get relevant INCAR for the current stage.
        self.ctx.vasp_base.vasp.parameters = get_stage_incar( #pylint: disable=unexpected-keyword-arg
            self.ctx.protocol, self.ctx.current_structure, orm.Str(self.ctx.stage_tag),
            hubbard_tag=self.ctx.hubbard_tag,
            prev_incar=self.ctx.prev_incar,
            modifications=orm.Dict(dict=self.ctx.modifications),
            metadata={
                'label':'get_stage_incar',
                'description': 'calcfuntion to get INCAR for current stage',
                'call_link_label':'run_get_stage_incar'
            })

        # Check and update input for kspacing
        if self.ctx.vasp_base.vasp.parameters.get_dict().get('KSPACING', None):
            self.inputs.kspacing = orm.Float(self.ctx.vasp_base.vasp.parameters.get_dict()['KSPACING'])
            self.inputs.kgamma = orm.Bool(self.ctx.vasp_base.vasp.parameters.get_dict().get('KSPACING', False))
            if self.inputs.kgamma:
                kpoints = set_kpoints( #pylint: disable=unexpected-keyword-arg
                        self.ctx.current_structure, self.inputs.kspacing, self.inputs.kgamma,
                        metadata={
                            'label':'set_kpoints',
                            'description': 'calcfuntion to construct kpoints from kspacing',
                            'call_link_label':'run_set_kpoints'
                        })
            else:
                kpoints = set_kpoints( #pylint: disable=unexpected-keyword-arg
                    self.ctx.current_structure, self.inpts.kspacing,
                    metadata={
                            'label':'set_kpoints',
                            'description': 'calcfuntion to construct kpoints from kspacing',
                            'call_link_label':'run_set_kpoints'
                    })
            self.ctx.vasp_base.vasp.kpoints = kpoints

        # Restart
        if self.ctx.restart_folder:
            self.ctx.vasp_base.vasp.restart_folder = self.ctx.restart_folder

        # Update lable anc call_link_label
        self.ctx.vasp_base.vasp['metadata'].update({
            'label':
            f'{self.ctx.stage_tag}_{self.ctx.stage_calc_types[self.ctx.stage_tag]}',
            'call_link_label':
            f'run_{self.ctx.stage_tag}_{self.ctx.stage_calc_types[self.ctx.stage_tag]}',
        })
        self.ctx.vasp_base['metadata'] = {}
        self.ctx.vasp_base['metadata']['label'] = self.ctx.vasp_base.vasp['metadata']['label']
        self.ctx.vasp_base['metadata']['call_link_label'] = self.ctx.vasp_base.vasp['metadata']['call_link_label']

        inputs = prepare_process_inputs(VaspBaseWorkChain, self.ctx.vasp_base)
        running = self.submit(VaspBaseWorkChain, **inputs)
        tag = self.ctx.stage_tag
        calc_type = self.ctx.stage_calc_types[self.ctx.stage_tag]
        self.report(f'Submitted VaspBaseWorkchain <pk>:{running.pk} for {tag}_{calc_type}')
        return ToContext(workchain=(append_(running)))

    def inspect_stage(self):  #pylint: disable=too-many-statements
        """Do the inspection of finished stage!"""
        try:
            workchain = self.ctx.workchain[-1]
        except IndexError:
            self.report(
                'There is no {} in the called workchain list.'.format(self.ctx.workchain_static[-1].process_label)
            )
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member

        if not workchain.is_finished_ok:
            self.report('Workchain failed with unrecoverable failure!')
            return self.exit_codes.ERROR_UNRECOVERABLE_FAILURE  # pylint: disable=no-member

        # We need to know if it is a production relax stage.
        self.ctx.prod_static = False
        self.ctx.prod_relax = False

        # Here we check of the run is production or burn! In case of production we need to check for convergence!
        if self.ctx.stage_calc_types[self.ctx.stage_tag] == 'relaxation':
            self.ctx.current_structure = workchain.outputs.structure
            if self.ctx.stage_tag != 'stage_0':
                self.ctx.prod_relax = True
            # In case someone just wants to run a single stage relax calculations.
            elif (self.ctx.stage_tag == 'stage_0' and len(list(self.ctx.protocol.get_dict().keys())) == 1):
                self.ctx.prod_relax = True

        if self.ctx.stage_calc_types[self.ctx.stage_tag] == 'static':
            if self.ctx.stage_tag != 'stage_0':
                self.ctx.prod_static = True
            # In case someone just wants to run a single stage static calculations.
            elif (self.ctx.stage_tag == 'stage_0' and len(list(self.ctx.protocol.get_dict().keys())) == 1):
                self.ctx.prod_static = True

        # If it is an inital stage, we do not look for convergence.
        if (not self.ctx.prod_static) and (not self.ctx.prod_relax):
            converged = True
        elif self.ctx.prod_static:
            converged = workchain.outputs.misc['converged_electronically']
        elif self.ctx.prod_relax:
            converged = workchain.outputs.misc['converged']

        # Assigning restart folder and INCAR from previous run
        self.ctx.restart_folder = workchain.outputs.remote_folder
        self.ctx.prev_incar = get_last_input(workchain)

        # Handling convergence issues for static and relax run.
        if (not converged) and self.ctx.prod_static:
            self.ctx.stage_iteration += 1
            # nelm = self.ctx.vasp_base.vasp.parameters.get_dict().get('NELM', 200) * 2
            nelm = self.ctx.prev_incar.get_dict().get('NELM', 200) * 2
            if self.ctx.vasp_base.vasp.parameters['ALGO'] in ['Fast', 'VeryFast']:
                self.ctx.modifications.update({'ALGO': 'Normal', 'NELM': nelm})
            elif self.ctx.vasp_base.vasp.parameters['ALGO'] in ['Normal']:
                self.ctx.modifications.update({'ALGO': 'All', 'NELM': nelm})
            # self.ctx.prev_incar = update_prev_incar(self.ctx.prev_incar, orm.Dict(dict=self.ctx.modifications))
            algo = self.ctx.modifications['ALGO']
            self.report(f'Electronic Convergence has not been reached: ALGO is set to {algo} and NELM is set to {nelm}')
        elif (not converged) and self.ctx.prod_relax:
            self.ctx.stage_iteration += 1
            # nsw = self.ctx.parameters.get_dict().get('NSW', 400) + 100
            nsw = self.ctx.prev_incar.get_dict().get('NSW', 400) + 100
            self.ctx.modifications.update({'NSW': nsw})
            # self.ctx.prev_incar = update_prev_incar(self.ctx.prev_incar, orm.Dict(dict=self.ctx.modifications))
            self.report(f'Ionic Convergence has not been reached: NSW is set to {nsw}')
        # If it is converged, we move on to next stage.
        elif converged:
            self.ctx.stage_idx += 1
            bg_down = workchain.outputs.misc['band_gap_spin_down']
            bg_up = workchain.outputs.misc['band_gap_spin_up']
            self.report(f'Band Gaps are {bg_down} and {bg_up}')
            self.out(
                f'final_incar.{self.ctx.stage_tag}_{self.ctx.stage_calc_types[self.ctx.stage_tag]}', self.ctx.prev_incar
            )
            self.ctx.all_outputs[f'{self.ctx.stage_tag}_{self.ctx.stage_calc_types[self.ctx.stage_tag]}'
                                 ] = workchain.outputs.misc
            self.ctx.stage_iteration = 0

        if self.ctx.stage_iteration > self.inputs.max_stage_iteration:
            self.report(f'Could not reach the convergence in stage_{self.ctx.stage_idx}! Better check them manually!')
            self.ctx.should_run_next_stage = False
            return self.exit_codes.ERROR_NON_CONVERGED_GEOMETRY  # pylint: disable=no-member

        if not f'stage_{self.ctx.stage_idx}' in self.ctx.protocol.get_dict():
            self.report('All stages are computed!')
            self.ctx.should_run_next_stage = False

    def results(self):
        """Attach the remaining output results."""
        self.out(
            'output_parameters',
            extract_wrap_results(
                **self.ctx.all_outputs,
                metadata={
                    'label': 'extract_wrap_results',
                    'description': 'calcfuntion to extract and wrap results of all stages.',
                    'call_link_label': 'run_extract_wrap_results'
                }
            )
        )

        self.out('structure', self.ctx.current_structure)
        self.report(
            'VaspMultiStageWorkChain is successfully finished! Output Dict <{}>'.format(
                self.outputs['output_parameters'].pk
            )
        )


# EOF
