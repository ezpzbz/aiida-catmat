"""
VaspMultiStageWorkChain
It is designed to perform  a flow of VASP calculations 
based on pre-defined requested stages. Current implementation
is tailored toward relaxation of structure by having an initial
static calculation and final static calculation to ensure 
electronic convergence as well as having accurate energies. 

Current implementation, takes the INCARs from Materials Project 
InputSets.
"""
import os

import yaml

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, ToContext, append_, while_, if_
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine.processes.exit_code import ExitCode

from aiida_bjm.calcfunctions import dict_merge
from aiida_bjm.utils import prepare_process_inputs
# from aiida_vasp.utils.aiida_utils import get_data_class
from pymatgen.io.vasp import sets as VaspInputSets
from pymatgen.io.vasp import Vasprun

VaspBaseWorkChain = WorkflowFactory('vasp.base') #pylint: disable=invalid-name
PotcarData = DataFactory('vasp.potcar')

@calcfunction
def get_potcar_mapping(structure, potcar_set_tag):
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(thisdir, '..', 'data', 'potcar_sets.yaml')
    
    with open(yaml_path) as default:
        potcar_dict = yaml.safe_load(default)
        sel_potcars = potcar_dict[potcar_set_tag.value]
        
    mapping = {}
    kinds = structure.get_kind_names()
    for kind in kinds:
        kind_no_digit = ''.join(i for i in kind if not i.isdigit())
        # This is workaround for structures with magnetic ordering.
        # Issue is that when we generate structure and assign initial
        # magnetic sites, they will be recongnized with different kinds.
        # Like: Fe1 and Fe2
        # We need to have separate entries for them in mapping so
        # aiida-vasp can construct the POTCAR
        # BUT after relaxation, the parsed structure does not carry same logic.
        # Although we have exact coverged magmoms for each site, these are not reflected
        # into the exported AiiDA structure object.
        # SO, when we try to run a consequent calculation on the relaxed structure,
        # We would face an excpetion saying that for example Fe is not provided in
        # mapping dict. 
        # Anyway, we provide exact magmoms for the conseuqnt calculations, we need to overcome this
        # issue. Either by having a function that assigns the magnetic sites on relax structure
        # which is not a good idea as it cannot handle them properly due to the aiida structure object
        # limitations. 
        # OR we can provide an entry in mapping dict for Fe itself from the beginning.
        mapping[kind_no_digit] = sel_potcars[kind_no_digit]
        mapping[kind] = sel_potcars[kind_no_digit]
    return orm.Dict(dict=mapping)

@calcfunction
def setup_protocols(protocol_tag, structure, user_incar_settings):
    """
    Reads stages from provided protocol file, and 
    constructs initial INCARs from Materials Project
    sets.
    """
    # Get pymathen structure object from AiiDA StructureData
    structure_pmg = structure.get_pymatgen_structure(add_spin=True)

    # Get user-defined stages and alternative settings from yaml file
    thisdir = os.path.dirname(os.path.abspath(__file__))
    protocol_path = os.path.join(thisdir, 'protocols', 'vasp', protocol_tag.value + '.yaml')
    with open(protocol_path, 'r') as protocol:
        protocol = yaml.safe_load(protocol)
    
    # User-defined INCAR settings passed to workchain.
    user_incar_settings = user_incar_settings.get_dict()
    
    def _update_magmoms(incar):
        magmom = incar['MAGMOM']
        for index, item in enumerate(magmom):
            if item == 0:
                magmom[index] = 0.6
            if item == -1:
                magmom[index] = -5
            if item == 1:
                magmom[index] = 5
        return magmom

    # # Get Hubbard parameters if DFT+U is requested
    def _set_ldau(structure, protocol):
        hubbard_dict = {
            'LDAU': True,
            'LDAUPRINT':1,
            'LDAUTYPE':2,
            'LMAXMIX':4
        }
        hubbard_params = protocol['hubbard']
        LDAUU = []
        LDAUJ = []
        LDAUL = []
        
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
        hubbard_dict.update({
            'LDAUU': LDAUU,
            'LDAUJ': LDAUJ,
            'LDAUL': LDAUL
        })
        protocol['ldau_section'] = hubbard_dict
        return protocol

    if user_incar_settings['LDAU']:
        protocol = _set_ldau(structure, protocol)
    # Get Input parameters for each stage.
    if protocol['stages']['initial_static'] or protocol['stages']['final_static']:
        input_set = getattr(VaspInputSets, 'MPStaticSet')
        inputs = input_set(structure_pmg, user_incar_settings=user_incar_settings)
        magmom = _update_magmoms(inputs.incar)
        if protocol['stages']['initial_static']:
            incar = inputs.incar
            incar.update({
                'LAECHG': False,
                'LCHARG': False,
                'LVHAR': False,
                'LWAVE': True,
                'EDIFF': 1e-4,
                'MAGMOM': magmom
            })
            protocol['incar_static_initial'] = {}
            if user_incar_settings['LDAU']:
                dict_merge(incar, protocol['ldau_section'])
                protocol['incar_static_initial'] = incar
            else:
                protocol['incar_static_initial'] = incar
        if protocol['stages']['initial_static']:
            incar = inputs.incar
            incar.update({
                'LAECHG': True,
                'LCHARG': True,
                'LVHAR': True,
                'LWAVE': True,
                'EDIFF': 1e-7,
                'MAGMOM': magmom
            })
            protocol['incar_static_final'] = {}
            if user_incar_settings['LDAU']:
                dict_merge(incar, protocol['ldau_section'])
                protocol['incar_static_final'] = incar
            else:
                protocol['incar_static_final'] = incar
    if protocol['stages']['relax']:
        input_set = getattr(VaspInputSets, 'MPRelaxSet')
        inputs = input_set(structure_pmg, user_incar_settings=user_incar_settings)
        incar = inputs.incar
        magmom = _update_magmoms(inputs.incar)
        incar.update({
            'LWAVE': True,
            'EDIFF': 1e-6,
            'MAGMOM': magmom
        })
        protocol['incar_relax'] = {}
        if user_incar_settings['LDAU']:
            dict_merge(incar, protocol['ldau_section'])
            protocol['incar_relax'] = incar
        else:
            protocol['incar_relax'] = incar
    if protocol['stages']['nscf']:
        protocol['incar_nscf'] = {}
        input_set = getattr(VaspInputSets, 'MPNonSCFSet')
        inputs = input_set(structure_pmg, user_incar_settings=user_incar_settings)
        protocol['incar_nscf'] = inputs.incar
                
    return orm.Dict(dict=protocol)

@calcfunction
def get_stage_incar(protocol, stage):
    d = protocol[stage.value]
    if stage.value == 'incar_relax' or stage.value == 'incar_static_final':
        d.update({
            'ISTART':1
        })
    return orm.Dict(dict=d)

@calcfunction
def increase_nsw(params):
    p = params.get_dict()
    p.update({
        'NSW':300
    })
    return orm.Dict(dict=p)

@calcfunction
def extract_wrap_results(**all_outputs):
    results_dict = {
        'energy_unit': 'eV'
    }
    if 'initial_static' in all_outputs.keys():
        energy = all_outputs['initial_static']['total_energies']['energy_no_entropy']
        results_dict['total_energy_initial_static'] = energy
    if 'relax_misc' in all_outputs.keys():
        energy = all_outputs['relax_misc']['total_energies']['energy_no_entropy']
        results_dict['total_energy_relax'] = energy
    if 'relax_mag' in all_outputs.keys():
        magmoms = []
        site_moms = all_outputs['relax_mag']['site_magnetization']['sphere']['x']['site_moment']
        tot_mag_site = all_outputs['relax_mag']['site_magnetization']['sphere']['x']['total_magnetization']['tot']
        tot_mag_full_cell = all_outputs['relax_mag']['site_magnetization']['full_cell']
        for v in site_moms.values():
            magmoms.append(v['tot'])
        results_dict['magmoms'] = magmoms
        results_dict['total_magnetizations_on_sites'] = tot_mag_site
        results_dict['total_magnetizations_full_cell'] = tot_mag_full_cell
    if 'final_static' in all_outputs.keys():
        energy = all_outputs['final_static']['total_energies']['energy_no_entropy']
        results_dict['total_energy_final_static'] = energy
        
    return orm.Dict(dict=results_dict)


class VaspMultiStageWorkChain(WorkChain):
    """Multi Stage Workchain"""
    _verbose = False

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            VaspBaseWorkChain, 
            include=['vasp.code', 'vasp.kpoints','vasp.restart_folder', 'vasp.metadata', 'vasp.potential'], namespace='vasp_base')
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.input('parameters', valid_type=orm.Dict)
        # spec.input('kpoints', valid_type=orm.KpointsData, required=False, help='The kpoints to use (KPOINTS).')
        spec.input('potential_family', valid_type=orm.Str, required=True)
        spec.input('potential_mapping', valid_type=orm.Dict, required=False)
        spec.input(
            'max_relax_iterations', 
            valid_type=orm.Int, 
            default=lambda: orm.Int(2))
        spec.input('settings', valid_type=orm.Dict, required=False)
        spec.input(
            'protocol_tag',
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str('standard'))
        spec.input(
            'potcar_set',
            valid_type=orm.Str,
            default=lambda: orm.Str('VASP'),
            help='Select which potcar set should be used to construct mappin. VASP or MPRelaxSet')
        # At some point, I will probably change the parser too.
        # I am still not convinved fully on using aiida-vasp parser.
        spec.input('vasp_base.vasp.metadata.options.parser_name',
                   valid_type=str,
                   default='vasp.vasp',
                   non_db=True,
                   help='Parser of the calculation: the default is cp2k_advanced_parser to get the necessary info')
        
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        # spec.exit_code(300,
        #                'ERROR_MISSING_REQUIRED_OUTPUT',
        #                message='the called workchain does not contain the necessary relaxed output structure')
        spec.exit_code(420, 'ERROR_NO_CALLED_WORKCHAIN', message='no called workchain detected')
        # spec.exit_code(500, 'ERROR_UNKNOWN', message='unknown error detected in the relax workchain')
        # spec.exit_code(502, 'ERROR_OVERRIDE_PARAMETERS', message='there was an error overriding the parameters')
        spec.outline(
            cls.initialize,
            if_(cls.should_run_initial_static)(
                cls.run_static,
                cls.verify_static,
                cls.inspect_static,
            ),
            while_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax
            ),
            if_(cls.should_run_final_static)(
                cls.run_static,
                cls.inspect_static,
            ),
            cls.results,
        )

        # spec.expose_outputs(VaspBaseWorkChain)
        spec.output('relax.structure', valid_type=orm.StructureData, required=False)
        spec.output('output_parameters', valid_type=orm.Dict, required=True)
        spec.output_namespace('final_incar', valid_type=orm.Dict, required=False, dynamic=True)
        

    def initialize(self):
        """Initialize."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.is_scf_converged = False
        self.ctx.is_strc_converged = False
        self.ctx.vasp_base = AttributeDict(self.exposed_inputs(VaspBaseWorkChain, 'vasp_base'))
        self.inputs.potential_mapping = get_potcar_mapping(self.ctx.current_structure, self.inputs.potcar_set)
        
        self.ctx.all_outputs = {}

        self.ctx.iteration = 0
    
        try:
            self._verbose = self.inputs.verbose.value
            # self.ctx.inputs.verbose = self.inputs.verbose
        except AttributeError:
            pass

        self.ctx.parameters = setup_protocols(
            self.inputs.protocol_tag, 
            self.ctx.current_structure,
            self.inputs.parameters)

        self.ctx.initial_static = self.ctx.parameters['stages']['initial_static']
        self.ctx.relax = self.ctx.parameters['stages']['relax']
        self.ctx.final_static = self.ctx.parameters['stages']['final_static']
        self.ctx.nscf = self.ctx.parameters['stages']['nscf']

        self._initialize_settings()

    def _initialize_settings(self):

        if 'settings' in self.inputs:
            settings = AttributeDict(self.inputs.settings.get_dict())
        else:
            settings = AttributeDict({'parser_settings': {}})
            settings = AttributeDict({'parser_settings': {}, 'ADDITIONAL_RETRIEVE_LIST':['INCAR']})
        
        if self.ctx.relax:
            dict_entry = {'add_structure': True}
            try:
                settings.parser_settings.update(dict_entry)
            except AttributeError:
                settings.parser_settings = dict_entry
        # self.ctx.inputs.settings = settings
        self.inputs.settings = settings
        
    def should_run_initial_static(self):
        """
        Perform initial static calculation to refine
        settings related to the electronic structure.
        """
        return self.ctx.initial_static

    def should_run_relax(self):
        """
        Run relaxation until it's converged or we reach the maximum allowed iteration.
        """
        return (not self.ctx.is_strc_converged) and (self.ctx.iteration < self.inputs.max_relax_iterations)
        
    def should_run_final_static(self):
        """
        We only wanna do it if we have a relaxed structure.
        """
        return self.ctx.is_strc_converged and self.ctx.final_static

    def run_static(self):
        """
        Prepares and submits static calculations as long as
        they are needed.
        """
        self.inputs.settings.parser_settings.add_structure = False
        
        self.ctx.vasp_base.vasp.structure = self.ctx.current_structure
        self.ctx.vasp_base.vasp.potential = PotcarData.get_potcars_from_structure(
                structure=self.inputs.structure,
                family_name=self.inputs.potential_family.value,
                mapping=self.inputs.potential_mapping.get_dict())
        if not self.ctx.is_strc_converged:
            self.ctx.vasp_base.vasp.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_static_initial'))
        else:
            self.ctx.vasp_base.vasp.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_static_final'))
            # self.ctx.inputs.settings.ADDITIONAL_RETRIEVE_LIST = ['INCAR']
        
        if self.ctx.vasp_base.vasp.parameters['ISPIN'] == 2:
            self.inputs.settings.parser_settings.add_site_magnetization = True
        self.ctx.vasp_base.vasp.settings = self.inputs.settings
        
        if self.ctx.is_strc_converged:
            self.ctx.vasp_base.vasp.restart_folder = self.ctx.workchain_relax[-1].outputs.remote_folder
            self.ctx.vasp_base.vasp['metadata'].update({
            'label': 'final_static',
            'call_link_label': 'final_static',
            })
        else:
            self.ctx.vasp_base['vasp']['metadata'].update({
            'label': 'initial_static',
            'call_link_label': 'initial_static',
            })
        
        inputs = prepare_process_inputs(VaspBaseWorkChain, self.ctx.vasp_base)
        running = self.submit(VaspBaseWorkChain, **inputs)
        self.report(f"Submitted Static VaspBaseWorkchain <pk>:{running.pk}")
        return ToContext(workchain_static=(append_(running)))

    
    def verify_static(self):
        """
        Inspects if static workchain finished ok 
        and also should check the scf convergence.
        """
        
        try:
            workchain = self.ctx.workchain_static[-1]
        except IndexError:
            self.report('There is no {} in the called workchain list.'.format(self.ctx.workchain_static[-1].process_label))
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member

        exit_status = workchain.exit_status
        exit_message = workchain.exit_message
        if not exit_status:
            self.report("Yei all good!")
            self.ctx.exit_code = self.exit_codes.NO_ERROR  # pylint: disable=no-member
        else:
            self.ctx.exit_code = ExitCode(exit_status, exit_message)
            self.report('The called {}<{}> returned a non-zero exit status. '
                        'The exit status {} is inherited'.format(workchain.__class__.__name__, workchain.pk, self.ctx.exit_code))
        return self.ctx.exit_code
    
    def inspect_static(self):

        workchain = self.ctx.workchain_static[-1]
        if not workchain.is_finished_ok:
            self.report("The initial static calculation did not finish properly")
            
        else:
            with workchain.outputs.retrieved.open('vasprun.xml') as xml:
                vasp_run = Vasprun(xml.name)
            self.ctx.is_scf_converged = vasp_run.converged_electronic
            if self.ctx.is_scf_converged:
                self.report('SCF converged')
                if self.ctx.is_strc_converged and self.ctx.final_static:
                    self.out('final_incar.final_static', workchain.inputs.vasp__parameters)
                    self.ctx.all_outputs['final_static'] = workchain.outputs.misc
                else:
                    self.out('final_incar.initial_static', workchain.inputs.vasp__parameters)
                    self.ctx.all_outputs['initial_static'] = workchain.outputs.misc
            else:
                self.ctx.is_scf_converged = False
             
    def run_relax(self):
        """
        Prepares and submit calculation to relax structure.
        """

        self.inputs.settings.parser_settings.add_structure = True
        self.inputs.structure = self.ctx.current_structure

        self.ctx.vasp_base.vasp.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_relax'))
        # if self.ctx.iteration == 0:
        #     self.ctx.inputs.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_relax'))
        # else:
        #     # Context is already for relax. We just wanna increase NSW.
        #     self.ctx.inputs.parameters = increase_nsw(self.ctx.inputs.parameters)

        if self.ctx.vasp_base.vasp.parameters['ISPIN'] == 2:
            self.inputs.settings.parser_settings.add_site_magnetization = True
                
        # Setting up remote folder for restart
        # if self.ctx.iteration == 0:
        #     self.ctx.inputs.restart_folder = self.ctx.workchain_static[-1].outputs.remote_folder
        # else:
        #     self.ctx.inputs.restart_folder = self.ctx.workchain_relax[-1].outputs.remote_folder
        if self.ctx.iteration == 0:
            self.ctx.vasp_base.vasp.restart_folder = self.ctx.workchain_static[-1].outputs.remote_folder
        else:
            self.ctx.vasp_base.vasp.restart_folder = self.ctx.workchain_relax[-1].outputs.remote_folder
        
        # self.ctx.inputs['metadata'].update({
        #     'label': f'relax_iteration_{self.ctx.iteration}',
        #     'call_link_label': f'relax_iteration_{self.ctx.iteration}',
        #     })
        
        self.ctx.vasp_base['vasp']['metadata'].update({
            'label': f'relax_iteration_{self.ctx.iteration}',
            'call_link_label': f'relax_iteration_{self.ctx.iteration}',
            })

        inputs = prepare_process_inputs(VaspBaseWorkChain, self.ctx.vasp_base)
        
        running = self.submit(VaspBaseWorkChain, **inputs)
        self.report(f"Submitted Relax VaspBaseWorkchain <pk>:{running.pk}_Iteration_{self.ctx.iteration}")
        self.ctx.iteration += 1
        return ToContext(workchain_relax=(append_(running)))    

    def inspect_relax(self):
        """
        Inspects if static workchain finished ok 
        and also should check the scf convergence.
        """

        workchain = self.ctx.workchain_relax[-1]
        if not workchain.is_finished_ok:
            self.report("Relax calculation did not finish properly")
        else:
            with workchain.outputs.retrieved.open('vasprun.xml') as xml:
                vasp_run = Vasprun(xml.name)
            self.ctx.is_strc_converged = vasp_run.converged
            if self.ctx.is_strc_converged:
                self.report(f'Fully converged in relax')
                self.ctx.relax = False
                self.out('final_incar.relax', workchain.inputs.vasp__parameters)
                self.ctx.all_outputs['relax_misc'] = workchain.outputs.misc
                self.ctx.all_outputs['relax_mag'] = workchain.outputs.site_magnetization
                self.ctx.current_structure = workchain.outputs.structure
            else:
                self.report("Structure is not converged!")
                
    def results(self):
        """Attach the remaining output results."""
        
        self.out('output_parameters', extract_wrap_results(**self.ctx.all_outputs))
        if self.ctx.is_strc_converged:
            self.out('relax.structure', self.ctx.current_structure)
        self.report('VaspMultiStageWorkChain finished successfull!')

        self.report('Output Dict <{}>'.format(self.outputs['output_parameters'].pk))
    