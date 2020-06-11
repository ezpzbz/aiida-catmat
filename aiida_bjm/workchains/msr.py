"""VaspMSRWorkChain: Multi Stage Relaxation WorkChain for VASP"""

import os
import yaml

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Dict, List, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_

VaspWorkChain = WorkflowFactory('vasp.vasp')  #pylint: disable=invalid-name

class VaspMSRWorkChain(WorkChain):
    """
    VaspMSRWorkChain is devised to read user-defined stages for Relaxation of structure 
    and performs them in order to get the relaxed structure.
    Inspired by MultiStageDdecWorkChain of CP2K:

    """

    @classmethod
    def define(cls, spec):
        super(VaspMSRWorkChain, cls).define(spec)

        spec.exposed_inputs(VaspWorkChain, namespace='vasp_base')
        spec.input('protocol_tag',
                   valid_type=Str,
                   default=lambda: Str('standard'),
                   required=False,
                   help='The tag of the protocol to be read from {tag}.yaml unless protocol_yaml input is specified')
        spec.input('protocol_modify',
                   valid_type=Dict,
                   default=lambda: Dict(dict={}),
                   required=False,
                   help='Specify custom settings that overvrite the yaml settings')
        spec.input('initial_settings_idx',
                   valid_type=Int,
                   default=lambda: Int(0),
                   required=False,
                   help='If idx>0 is chosen, jumps directly to overwrite settings_0 with settings_{idx}')      
        spec.outline(
            
        )

        spec.expose_outputs(VaspWorkChain)
        
    def initialize(self):
        """Initialize."""
        self._init_context()
        self._init_inputs()

    def _init_context(self):
        """Initialize context variables that are used during the logical flow."""
        self.ctx.is_finished = False
        self.ctx.iteration = 0
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspWorkChain, 'vasp_base'))

        self.ctx.is_settings_ok = False
    
    def setup_multistage(self):
        
        self.ctx.restart_folder = self.inputs.restart_folder if 'restart_folder' in self.inputs else None        
        # Read protocol
        thisdir = os.path.dirname(os.path.abspath(__file__))
        yamlfullpath = os.path.join(thisdir, 'protocols', 'vasp', self.inputs.protocol_tag.value + '.yaml')
        with open(yamlfullpath, 'r') as stream:
            self.ctx.protocol = yaml.safe_load(stream)
        dict_merge(self.ctx.protocol, self.inputs.protocol_modify.get_dict())
    
    def should_run_stage0(self):
        return not self.ctx.is_settings_ok
    
    def run_stage(self):
        
        # In the first implementation we just have one setting.
        # It will be used to run stage0 (ISIF=2 and NSW=5)
        # Then, the WAVECAR and structure will be used for ISIF=3.
        # This will be changed in near furture by also exploring the 
        # effect of setting on the SCF convergence. It will become
        # very handy when we need to change settings to improve the
        # SCF convergence.




        running = self.submit(VaspWorkChain, **self.ctx.inputs)
        self.report("submitted VaspWorkChain for {}/{}".format(self.ctx.stage_tag, self.ctx.settings_tag))
        return ToContext(stages=append_(running))

    def inspect_and_update_settings_stage0(self):
        self.ctx.is_settings_ok = True

        # Herein, we need to check the output of setting_0/stage_0
        # to check if SCF is converged:

        if 'output_parameters' in self.ctx.stages[-1].outputs:
            output_dict = self.ctx.stages[-1].outputs.output_parameters
        else:
            self.report('ERROR_PARSING_OUTPUT')
            return self.exit_codes.ERROR_PARSING_OUTPUT  # pylint: disable=no-member

        if not output_dict['scf_converged']:
            self.ctx.is_settings_ok = False

        if not self.ctx.is_settings_ok:
            next_settings_tag = 'settings_{}'.format(self.ctx.settings_idx)
            if next_settings_tag in self.ctx.protocol:
                self.ctx.settings_tag = next_settings_tag
                dict_merge(self.ctx.params, self.ctx.protocol[self.ctx.settings_tag])
            else:
                return self.exit_codes.ERROR_NO_MORE_SETTINGS  # pylint: disable=no-member
    
    def inspect_and_update_stage(self):
        # Here we will have the structure update 
        # and linking to the remote folder that contains
        # previous WAVECAR
    
    def should_run_stage(self):
        return self.ctx.next_stage_exists
    
    def results(self):
        for index, stage in enumerate(self.ctx.stages):
            all_output_parameters[f'output_parameters_{index}'] = stage.outputs.output_parameters
        
        # write calcfuntion to extract and put all relevant outputs in one Dict object

        # Get and report the latest input parameters, so it can be used as starting point 
        # for potential future calculations.

        # Output the relaxed structure.

        






