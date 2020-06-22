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
from aiida.plugins import WorkflowFactory

from aiida_bjm.calcfunctions import dict_merge, aiida_dict_merge
from aiida_bjm.utils import prepare_process_inputs

from pymatgen.io.vasp import sets as VaspInputSets
from pymatgen.io.vasp import Vasprun

VaspBaseWorkChain = WorkflowFactory('vasp.vasp') #pylint: disable=invalid-name

@calcfunction
def setup_protocols(protocol_tag, structure, user_incar_settings):
    """
    Reads stages from provided protocol file, and 
    constructs initial INCARs from Materials Project
    sets.
    """
    # Get pymathen structure object from AiiDA StructureData
    structure = structure.get_pymatgen_structure(add_spin=True)

    # Get user-defined stages and alternative settings from yaml file
    thisdir = os.path.dirname(os.path.abspath(__file__))
    protocol_path = os.path.join(thisdir, 'protocols', 'vasp', protocol_tag.value + '.yaml')
    with open(protocol_path, 'r') as protocol:
        protocol = yaml.safe_load(protocol)
    
    # User-defined INCAR settings passed to workchain.
    user_incar_settings = user_incar_settings.get_dict()
    
    # Get Hubbard parameters if DFT+U is requested
    if 'LDAU' in user_incar_settings:
        if user_incar_settings['LDAU']:
            hubbard_dict = {'LDAUU':{}}
            hubbard_params = protocol['hubbard']['LDAUU']
            symbol_set = structure.symbol_set
            for symb in symbol_set:
                if symb in hubbard_params:
                    hubbard_dict['LDAUU'][symb] = hubbard_params[symb]
        # Update user-defined INCAR settings with Hubbard parameters
            dict_merge(user_incar_settings, hubbard_dict)
        
    # Get Input parameters for each stage.
    # TODO: logic needs to improved. 
    if protocol['stages']['initial_static'] or protocol['stages']['final_static']:
        protocol['incar_static'] = {}
        input_set = getattr(VaspInputSets, 'MPStaticSet')
        inputs = input_set(structure, user_incar_settings=user_incar_settings)
        if protocol['stages']['initial_static']:
            incar = inputs.incar
            magmom = incar['MAGMOM']
            for index, item in enumerate(magmom):
                if item == 0:
                    magmom[index] = 0.6
                if item == -1:
                    magmom[index] = -5
                if item == 1:
                    magmom[index] = 5
            incar.update({
                'LAECHG': False,
                'LCHARG': False,
                'LVHAR': False,
                'EDIFF': 1e-4,
                'MAGMOM': magmom
            })
            protocol['incar_static'] = incar
        else:
            # Same as above
            incar = inputs.incar
            magmom = incar['MAGMOM']
            for index, item in enumerate(magmom):
                if item == 0:
                    magmom[index] = 0.6
                if item == -1:
                    magmom[index] = -5
                if item == 1:
                    magmom[index] = 5
            incar.update({
                'LAECHG': False,
                'LCHARG': False,
                'LVHAR': False,
                'EDIFF': 1e-7,
                'PREC': 'Accurate',
                'MAGMOM': magmom 
            })
            protocol['incar_static'] = incar
    if protocol['stages']['relax']:
        protocol['incar_relax'] = {}
        input_set = getattr(VaspInputSets, 'MPRelaxSet')
        inputs = input_set(structure, user_incar_settings=user_incar_settings)
        incar = inputs.incar
        magmom = incar['MAGMOM']
        for index, item in enumerate(magmom):
            if item == 0:
                magmom[index] = 0.6
            if item == -1:
                magmom[index] = -5
            if item == 1:
                magmom[index] = 5
        incar.update({
            'LWAVE': True,
            'EDIFF': 1e-6,
            'PREC': 'Accurate',
            'ADDGRID': True,
            'MAGMOM': magmom
        })
        protocol['incar_relax'] = incar
    if protocol['stages']['nscf']:
        protocol['incar_nscf'] = {}
        input_set = getattr(VaspInputSets, 'MPNonSCFSet')
        inputs = input_set(structure, user_incar_settings=user_incar_settings)
        protocol['incar_nscf'] = inputs.incar
            
    return orm.Dict(dict=protocol)

@calcfunction
def get_stage_incar(protocol, stage):
    return orm.Dict(dict=protocol[stage.value])

class VaspMultiStageWorkChain(WorkChain):
    """Multi Stage Workchain"""
    _verbose = False

    @classmethod
    def define(cls, spec):
        super(VaspMultiStageWorkChain, cls).define(spec)
        spec.expose_inputs(
            VaspBaseWorkChain, 
            exclude=('parameters', 'structure', 'settings'), namespace='base')
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.input('parameters', valid_type=orm.Dict)
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
        
        # spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        # spec.exit_code(300,
        #                'ERROR_MISSING_REQUIRED_OUTPUT',
        #                message='the called workchain does not contain the necessary relaxed output structure')
        # spec.exit_code(420, 'ERROR_NO_CALLED_WORKCHAIN', message='no called workchain detected')
        # spec.exit_code(500, 'ERROR_UNKNOWN', message='unknown error detected in the relax workchain')
        # spec.exit_code(502, 'ERROR_OVERRIDE_PARAMETERS', message='there was an error overriding the parameters')
        spec.outline(
            cls.initialize,
            if_(cls.should_run_initial_static)(
                cls.run_static,
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

        spec.expose_outputs(VaspBaseWorkChain)
        spec.output('relax.structure', valid_type=orm.StructureData, required=False)
        spec.output_namespace('final_incar', valid_type=orm.Dict, required=False, dynamic=True)
        

    def initialize(self):
        """Initialize."""
        self.ctx.current_structure = self.inputs.structure
        self.ctx.is_scf_converged = False
        self.ctx.is_strc_converged = False
        self.ctx.inputs = AttributeDict()

        self.ctx.iteration = 0
    
        try:
            self._verbose = self.inputs.verbose.value
            self.ctx.inputs.verbose = self.inputs.verbose
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

        # self.ctx.settings_ok = False
        # self.ctx.static_stage_idx = 0
        # self.ctx.relax_stage_idx = 0
        # self.ctx.static_stage_tag = f'relax_stage_{self.ctx.static_stage_idx}'
        # self.ctx.relax_stage_tag = f'relax_stage_{self.ctx.relax_stage_idx}'
        # self.ctx.settings_idx = 0
        # self.ctx.settings_tag = f'settings_{self.ctx.settings_idx}'

    def _initialize_settings(self):

        if 'settings' in self.inputs:
            settings = AttributeDict(self.inputs.settings.get_dict())
        else:
            settings = AttributeDict({'parser_settings': {}})
        
        if self.ctx.relax:
            dict_entry = {'add_structure': True}
            try:
                settings.parser_settings.update(dict_entry)
            except AttributeError:
                settings.parser_settings = dict_entry
        self.ctx.inputs.settings = settings
    
    def should_run_initial_static(self):
        """
        Perform initial static calculation to refine
        settings related to the electronic structure.
        """
        return self.ctx.initial_static
        # return not self.ctx.is_scf_converged

    def should_run_relax(self):
        """
        Run relaxation until it's converged or we reach the maximum allowed iteration.
        """
        return self.ctx.relax
        # return self.ctx.relax and not self.ctx.is_scf_converged and self.ctx.iteration < self.inputs.max_relax_iterations
    
    def should_run_final_static(self):
        """
        We want to have it on relaxed structure.
        Later, we may have it optional.
        """
        return self.ctx.final_static

    def run_static(self):
        """
        Prepares and submits static calculations as long as
        they are needed.
        """
        self.ctx.inputs.settings.parser_settings.add_structure = False
        self.ctx.inputs.structure = self.ctx.current_structure
        
        self.ctx.inputs.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_static'))

        self.ctx.inputs.update(self.exposed_inputs(VaspBaseWorkChain, 'base'))
        
        if self.ctx.is_strc_converged:
            self.ctx.inputs.restart_folder = self.ctx.workchain_relax[-1].outputs.remote_folder
        
        inputs = prepare_process_inputs(VaspBaseWorkChain, self.ctx.inputs)
        
        running = self.submit(VaspBaseWorkChain, **inputs)
        self.report("Submmited Static VaspBaseWorkchain <pk> {}".format(running.pk))
        return ToContext(workchain_static=(append_(running)))    
    
    def inspect_static(self):
        """
        Inspects if static workchain finished ok 
        and also should check the scf convergence.
        """

        workchain = self.ctx.workchain_static[-1]
        if not workchain.is_finished_ok:
            self.report("Something was wrong!")
        else:
            with workchain.outputs.retrieved.open('vasprun.xml') as xml:
                vasp_run = Vasprun(xml.name)
            converged = vasp_run.converged
            converged_electronic = vasp_run.converged_electronic
            converged_ionic = vasp_run.converged_ionic
            self.report(f'Fully converged {converged}')
            self.report(f'SCF converged {converged_electronic}')
            self.report(f'Ionic converged {converged_ionic}')
            if self.ctx.is_strc_converged:
                self.out('final_incar.final_static', workchain.inputs.parameters)
            else:
                self.out('final_incar.initial_static', workchain.inputs.parameters)
             
    def run_relax(self):
        """
        Prepares and submit calculation to relax structure.
        """

        self.ctx.inputs.settings.parser_settings.add_structure = True
        self.ctx.inputs.structure = self.ctx.current_structure

        self.ctx.inputs.parameters = get_stage_incar(self.ctx.parameters, orm.Str('incar_relax'))

        self.ctx.inputs.update(self.exposed_inputs(VaspBaseWorkChain, 'base'))
        self.ctx.inputs.restart_folder = self.ctx.workchain_static[-1].outputs.remote_folder
        

        inputs = prepare_process_inputs(VaspBaseWorkChain, self.ctx.inputs)
        
        running = self.submit(VaspBaseWorkChain, **inputs)
        self.report("Submmited Relax VaspBaseWorkchain <pk> {}".format(running.pk))
        return ToContext(workchain_relax=(append_(running)))    

    def inspect_relax(self):
        """
        Inspects if static workchain finished ok 
        and also should check the scf convergence.
        """

        workchain = self.ctx.workchain_relax[-1]
        if not workchain.is_finished_ok:
            self.report("Something was wrong!")
        else:
            with workchain.outputs.retrieved.open('vasprun.xml') as xml:
                vasp_run = Vasprun(xml.name)
            converged = vasp_run.converged
            converged_electronic = vasp_run.converged_electronic
            converged_ionic = vasp_run.converged_ionic
            self.report(f'Fully converged in relax {converged}')
            self.report(f'SCF converged in relax {converged_electronic}')
            self.report(f'Ionic converged in relax {converged_ionic}')
            self.ctx.is_strc_converged = True
            self.ctx.relax = False
            self.ctx.current_structure = workchain.outputs.structure
            self.out('final_incar.relax', workchain.inputs.parameters)

    def results(self):
        """Attach the remaining output results."""

        workchain = self.ctx.workchain_static[-1]
        workchain_relax = self.ctx.workchain_relax[-1]
        relaxed_structure = workchain_relax.outputs.structure
        self.out('relax.structure', relaxed_structure)
        self.out_many(self.exposed_outputs(workchain, VaspBaseWorkChain))
    