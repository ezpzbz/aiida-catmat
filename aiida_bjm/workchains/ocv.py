"""
VaspOcvWorkChain: VASP workchain that wraps two consecutive relax workchains.
It starts with relaxing lithiated structure.
Takes the relaxed structure.
Removes the Li ions
Modifies the input. 
Relax the delithiated structure
calculates the OCV
"""

# import os
# import yaml

# AiiDA modules
from aiida.plugins import DataFactory, WorkflowFactory
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain
# , append_, if_, while_

# import pymatgen as pmg

StructureData = DataFactory('structure') #pylint: disable=invalid-name


VaspRelaxWorkChain = WorkflowFactory('vasp.relax')  #pylint: disable=invalid-name

DEFAULT_PARAMETERS = {}

# calcfunctions in order of appearance in code
@calcfunction
def modify_structure(relaxed_structure, to_remove):
    """
    gets the relaxed structur from previous workchain,
    removes the Li ions
    returns the StructureData for next workchain
    """
    pmg_strc = relaxed_structure.get_pymatgen_structure()
    pmg_strc.remove(to_remove.value)
    return StructureData(pymatgen_structure=pmg_strc)

@calcfunction
def calculate_ocv(output1,output2,strc1,strc2,ion, mu):
    """
    Gets the output dictionary of two structures
    also we need to provide structures and ion
    Calculates the OCV
    returns the result in new Dict object
    """
    out_dict1 = output1.get_dict()
    out_dict2 = output2.get_dict()
    strc1 = strc1.get_pymatgen_structure()
    strc2 = strc2.get_pymatgen_structure()
    # Count the difference in the number of specified ion in two structures
    # number_of_ion = 2 
    ocv = -(out_dict1['final_energy'] - out_dict2['final_energy'] - number_of_ion * mu) / number_of_ion
    return orm.Float(ocv)


class VaspOcvWorkChain(WorkChain):
    """
    VaspOcvWorkChain is devised to read user-defined stages for Relaxation of structure 
    and performs them in order to get the relaxed structure.
    Inspired by MultiStageDdecWorkChain of CP2K:

    """

    @classmethod
    def define(cls, spec):
        super(VaspOcvWorkChain, cls).define(spec)

        spec.expose_inputs(
            VaspRelaxWorkChain, namespace='vasp_relax',
            exclude=['parameters']
            )
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict=DEFAULT_PARAMETERS),
            required=False,
            help='Parameters that are needed to setup the calculations')
        
        # Minimal outline to have it running. 
        # TODO: adding another stage or stages for reperforming
        # workchains on delithiated structures.
        spec.outline(
            cls.initialize,
            cls.run_relax,
            cls.return_results
        )

        spec.expose_outputs(VaspRelaxWorkChain)
        
    def initialize(self):
        """Initialize."""
        # self._init_context()
        # self._init_inputs()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspRelaxWorkChain, 'vasp_base'))
    # def _init_context(self):
    #     """Initialize context variables that are used during the logical flow."""
    #     self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspRelaxWorkChain, 'vasp_base'))
        self.ctx.params = self.input.parameters
    def run_relax(self):
        """Run relax workchain """
        self.ctx.inputs['vasp_relax']['parameters'] = self.ctx.params 

        running = self.submit(VaspRelaxWorkChain, **self.ctx.inputs)
        self.report("submitted VaspRelaxWorkChain")
        return ToContext(stages=running)
        
    def return_results(self):
        """return results """
        
        self.out(self.exposed_outputs(VaspRelaxWorkChain))
        
        # write calcfuntion to extract and put all relevant outputs in one Dict object

        # Get and report the latest input parameters, so it can be used as starting point 
        # for potential future calculations.

        # Output the relaxed structure.

        






