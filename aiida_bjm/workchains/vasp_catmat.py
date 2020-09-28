"""VaspCatMatWorkChain
It wraps VaspMultiStageWorkChain to perform two ccnsecutive
calculation for calculation of open circuit voltage.
"""
from pymatgen import Element

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, ToContext, if_
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
def calculate_ocv(discharged, charged, discharged_structure, charged_structure, anode):  #pylint: disable=too-many-locals
    """Calculate OCV"""
    F_CONSTANT = 96485.3  #C/mol #pylint: disable=invalid-name

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

    discharged_structure_pmg = discharged_structure.get_pymatgen_structure()
    charged_structure_pmg = charged_structure.get_pymatgen_structure()

    a_dischg = discharged_structure_pmg.lattice.a
    b_dischg = discharged_structure_pmg.lattice.b
    c_dischg = discharged_structure_pmg.lattice.c
    vol_dischg = discharged_structure_pmg.lattice.volume

    a_chg = charged_structure_pmg.lattice.a
    b_chg = charged_structure_pmg.lattice.b
    c_chg = charged_structure_pmg.lattice.c
    vol_chg = charged_structure_pmg.lattice.volume

    a_change = ((a_chg - a_dischg) / a_dischg) * 100
    b_change = ((b_chg - b_dischg) / b_dischg) * 100
    c_change = ((c_chg - c_dischg) / c_dischg) * 100
    vol_change = ((vol_chg - vol_dischg) / vol_dischg) * 100

    ocv = -((enrg_discharged - enrg_charged - (anode_mu * nions)) / nions)
    density_dischg = discharged_structure_pmg.density  #g/cm3
    mw_dischg = discharged_structure_pmg.composition.weight  #g/mol
    specific_capacity_grav = (nions * F_CONSTANT * 1000) / (mw_dischg * 3600)  #mAh/g
    specific_capacity_vol = ((nions * F_CONSTANT * 1000) / (mw_dischg * 3600)) * density_dischg  #mAh/cm3
    energy_density_grav = specific_capacity_grav * ocv  #Wh/kg
    energy_density_vol = specific_capacity_vol * ocv  #Wh/L

    ocv_dict = {
        'energy_unit': 'eV',
        'voltage_unit': 'V',
        'lattice_abc_unit': 'A',
        'volume_unit': 'A^3',
        'gravimetric_specific_capacity_unit': 'mAh/g',
        'volumetric_specific_capacity_unit': 'mAh/cm^3',
        'gravimetric_energy_density_unit': 'Wh/kg',
        'volumetric_energy_density_unit': 'Wh/L',
        'lattice_a_change': round(a_change, 3),
        'lattice_b_change': round(b_change, 3),
        'lattice_c_change': round(c_change, 3),
        'volume_change': round(vol_change, 1),
        'ocv': round(ocv, 2),
        'gravimetric_specific_capacity': round(specific_capacity_grav, 1),
        'volumetric_specific_capacity': round(specific_capacity_vol, 1),
        'gravimetric_energy_density': round(energy_density_grav, 1),
        'volumetric_energy_density': round(energy_density_vol, 1),
        'energy_of_charged_state': enrg_charged,
        'energy_of_discharged_state': enrg_discharged,
        'num_of_ions': nions,
        'anode_mu': anode_mu
    }
    return orm.Dict(dict=ocv_dict)


class VaspCatMatWorkChain(WorkChain):
    """Convergence WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Expose VaspMultiStageWorkChain inputs
        spec.expose_inputs(VaspMultiStageWorkChain)

        # Define VaspCatMatWorkChain specific inputs
        spec.input('anode', valid_type=orm.Dict, required=True, help='Chemical potential of anode')
        spec.input(
            'discharged_calculated_data',
            valid_type=orm.Dict,
            required=False,
            help='Output dictionary of previously calculated structure!'
        )
        spec.input(
            'discharged_relaxed_structure',
            valid_type=StructureData,
            required=False,
            help='Relaxed structure of previously calculated structure!'
        )

        # Exit codes

        # Define outline
        spec.outline(
            cls.initialize,
            if_(cls.should_run_discharged)(cls.run_discharged),
            cls.run_charged,
            cls.results,
        )
        # Expose outputs
        # spec.expose_outputs(VaspMultiStageWorkChain)
        spec.output('ocv_results', valid_type=orm.Dict, required=False, help='output dictionary with OCV results')

    def initialize(self):
        """Initialize inputs and settings"""
        # Setup inputs
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspMultiStageWorkChain))

    def should_run_discharged(self):
        """Checks whether should run calculation on discharged structure"""
        self.ctx.should_run_discharged = True
        if 'discharged_calculated_data' in self.inputs:
            if 'discharged_relaxed_structure' in self.inputs:
                self.ctx.discharged_out = self.inputs.discharged_calculated_data
                self.ctx.discharged_strc = self.inputs.discharged_relaxed_structure
        return self.ctx.should_run_discharged

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
        if self.ctx.should_run_discharged:
            self.ctx.inputs.structure = update_structure(self.ctx.wc_discharged.outputs.structure, self.inputs.anode)
        else:
            self.ctx.inputs.structure = update_structure(self.ctx.discharged_relaxed_structure, self.inputs.anode)
        self.ctx.inputs['metadata']['label'] = 'charged_structured'
        self.ctx.inputs['metadata']['call_link_label'] = 'run_charged_structured'
        running = self.submit(VaspMultiStageWorkChain, **self.ctx.inputs)
        self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for charged structure!')
        return ToContext(wc_charged=running)

    def results(self):
        """Handle results"""
        if self.ctx.should_run_discharged:
            self.ctx.discharged_calculated_data = self.ctx.wc_discharged.outputs.output_parameters
            self.ctx.charged_calculated_data = self.ctx.wc_charged.outputs.output_parameters
            self.ctx.discharged_relaxed_structure = self.ctx.wc_discharged.outputs.structure
            self.ctx.charged_relaxed_structure = self.ctx.wc_charged.outputs.structure
        else:
            self.ctx.charged_calculated_data = self.ctx.wc_charged.outputs.output_parameters
            self.ctx.charged_relaxed_structure = self.ctx.wc_charged.outputs.structure
        self.out(
            'ocv_results',
            calculate_ocv(
                self.ctx.discharged_calculated_data, self.ctx.charged_calculated_data,
                self.ctx.discharged_relaxed_structures, self.ctx.scharged_relaxed_structures, self.inputs.anode
            )
        )
        self.report('VaspCatMatWorkChain FInished Successfully!')


# EOF
