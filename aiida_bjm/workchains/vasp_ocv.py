"""VaspOcvWorkChain
It wraps VaspMultiStageWorkChain to perform two ccnsecutive
calculation for calculation of open circuit voltage.
"""
from pymatgen import Element

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, ToContext
from aiida.plugins import DataFactory, WorkflowFactory

VaspMultiStageWorkChain = WorkflowFactory('bjm.vasp_multistage')  #pylint: disable=invalid-name
PotcarData = DataFactory('vasp.potcar')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name
StructureData = DataFactory('structure')  #pylint: disable=invalid-name


@calcfunction
def update_structure(structure, anode):
    """Update structure by removing ions (Li/Na/...)"""
    strc_pmg = structure.get_pymatgen_structure(add_spin=True)
    species = strc_pmg.species
    el_to_remove = list(anode.get_dict().keys())
    for sp in species:  #pylint: disable=invalid-name
        if sp.element == Element(el_to_remove[0]):
            strc_pmg.remove_species([sp])
    return StructureData(pymatgen_structure=strc_pmg)


@calcfunction
def calculate_ocv(discharged, charged, discharged_structure, anode):
    """Calculate OCV"""

    def _get_stg_idx(item):
        return item.split('_')[1]

    stgs = list(discharged.keys())
    stgs.sort(key=_get_stg_idx)
    last_stg = stgs[-1]
    enrg_discharged = discharged.get_dict()[last_stg]['final_energy']
    enrg_charged = charged.get_dict()[last_stg]['final_energy']
    anode_info = anode.get_dict()
    anode_el = list(anode_info.keys())[0]
    anode_mu = list(anode_info.values())[0]
    nions = discharged_structure.get_composition()[anode_el]
    ocv = -(enrg_discharged - enrg_charged - (anode_mu * nions) / nions)
    ocv_dict = {
        'energy_unit': 'eV',
        'voltage_unit': 'V',
        'ocv': ocv,
        'energy_of_charged_state': enrg_charged,
        'energy_of_discharged_state': enrg_discharged,
        'num_of_ions': nions,
        'anode_mu': anode_mu
    }
    return orm.Dict(dict=ocv_dict)


class VaspOcvWorkChain(WorkChain):
    """Convergence WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Expose VaspMultiStageWorkChain inputs
        spec.expose_inputs(VaspMultiStageWorkChain)

        # Define VaspOcvWorkChain specific inputs
        spec.input('anode', valid_type=orm.Dict, required=True, help='Chemical potential of anode')

        # Exit codes

        # Define outline
        spec.outline(
            cls.initialize,
            cls.run_discharged,
            cls.run_charged,
            cls.results,
        )
        # Expose outputs
        spec.expose_outputs(VaspMultiStageWorkChain)
        spec.output('ocv_results', valid_type=orm.Dict, required=False, help='output dictionary with OCV results')

    def initialize(self):
        """Initialize inputs and settings"""
        # Setup inputs
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspMultiStageWorkChain))

    def run_discharged(self):
        """Submit VaspMultiStageWorkChain on discharged structure"""
        self.ctx.inputs['metadata']['label'] = 'discharged_structured'
        self.ctx.inputs['metadata']['call_link_label'] = 'run_discharged_structured'
        running = self.submit(VaspMultiStageWorkChain, **self.ctx.inputs)
        self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for discharged structure!')
        return ToContext(wc_discharged=running)

    def run_charged(self):
        """Submit VaspMultiStageWorkChain on charged structure"""
        # Update structure
        self.ctx.inputs.structure = update_structure(self.ctx.wc_discharged.outputs.structure, self.inputs.anode)
        self.ctx.inputs['metadata']['label'] = 'charged_structured'
        self.ctx.inputs['metadata']['call_link_label'] = 'run_charged_structured'
        running = self.submit(VaspMultiStageWorkChain, **self.ctx.inputs)
        self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for charged structure!')
        return ToContext(wc_charged=running)

    def results(self):
        """Handle results"""
        discharged_out = self.ctx.wc_discharged.outputs.output_parameters
        charged_out = self.ctx.wc_charged.outputs.output_parameters
        discharged_strc = self.ctx.wc_discharged.outputs.structure
        self.out('ocv_results', calculate_ocv(discharged_out, charged_out, discharged_strc, self.inputs.anode))
        self.report('VaspOcvWorkChain FInished Successfully!')


# EOF
