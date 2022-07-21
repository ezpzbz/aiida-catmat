"""`VaspConvergeWorkChain` that wraps VaspMultiStageWorkChain to series of static calculations for
getting converged ENCUT and KSPACING
"""

from aiida.orm import Dict, Float, Int, List
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, while_
from aiida.plugins import DataFactory, WorkflowFactory

VaspMultiStageWorkChain = WorkflowFactory('catmat.vasp_multistage')  #pylint: disable=invalid-name
PotcarData = DataFactory('vasp.potcar')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name


@calcfunction
def update_incar_encut(parameters: Dict, encut: Int) -> Dict:
    """Updates the `ENCUT` tag in `INCAR`

    Args:
        parameters (Dict): `INCAR` dictionary
        encut (Int): The new `ENCUT` value

    Returns:
        Dict: Updated `INCAR` with new `ENCUT`
    """
    param = parameters.get_dict()
    param.update({'ENCUT': encut.value})
    return Dict(dict=param)


@calcfunction
def identify_encut_convergence(threshold: Float, **all_encut_outputs: Dict) -> Dict:
    """Reports the energies for different `ENCUT` values and identifies the converged one

    Args:
        threshold (Float): The threshold for energy changes to consider a `ENCUT` converged.

    Returns:
        Dict: Results dictionary where `converged_encut` is the identified `ENCUT`. The `converged_encut_conservative`
        is one step beyond the `converged_encut`.
    """
    results = {}
    results['final_energy'] = {}
    results['final_energy_per_atom'] = {}

    for key, value in all_encut_outputs.items():
        encut = int(key.split('_')[1])
        results['final_energy'][encut] = value['stage_0_static']['final_energy']
        results['final_energy_per_atom'][encut] = value['stage_0_static']['final_energy_per_atom']

    encut_list = list(results['final_energy'].keys())
    for en1, en2 in zip(encut_list, encut_list[1:]):
        dE = abs(results['final_energy_per_atom'][en2] - results['final_energy_per_atom'][en1])  #pylint: disable=invalid-name
        if dE < threshold.value:  #pylint: disable=no-else-break
            results['converged_encut'] = en1
            results['converged_encut_conservative'] = en2
            results['energy_difference'] = dE
            break
        else:
            results['converged_encut'
                    ] = f'Sorry! Energy difference <{dE}> is still above threshold <{threshold.value}>!'
    return Dict(dict=results)


@calcfunction
def identify_kspacing_convergence(threshold: Float, **all_kspacing_outputs: Dict) -> Dict:
    """Reports energies for different KSPACING and identifies the converged one

    Args:
        threshold (Float): The threshold for energy changes to consider a `KSPACING` converged.

    Returns:
        Dict: Results dictionary where `converged_kspacing` is the identified `ENCUT`.
        The `converged_kspacing_conservative` is one step beyond the `converged_encut`.
    """
    results = {}
    results['final_energy'] = {}
    results['final_energy_per_atom'] = {}

    for key, value in all_kspacing_outputs.items():
        kspacing = float(key.split('_')[1]) / 1000
        results['final_energy'][kspacing] = value['stage_0_static']['final_energy']
        results['final_energy_per_atom'][kspacing] = value['stage_0_static']['final_energy_per_atom']

    kspacing_list = list(results['final_energy'].keys())
    kspacing_list.sort(reverse=True)
    for en1, en2 in zip(kspacing_list, kspacing_list[1:]):
        dE = abs(results['final_energy_per_atom'][en2] - results['final_energy_per_atom'][en1])  #pylint: disable=invalid-name
        if dE < threshold.value:  #pylint: disable=no-else-break
            results['converged_kspacing'] = en1
            results['converged_kspacing_conservative'] = en2
            results['energy_difference'] = dE
            break
        else:
            results['converged_kspacing'
                    ] = f'Sorry! Energy difference <{dE}> is still above threshold <{threshold.value}>!'
    return Dict(dict=results)


@calcfunction
def return_final_results(encut_results: Dict, kpspacing_results: Dict) -> Dict:
    """Returns a single dict with converged `ENCUT` and `KSPACING`

    Args:
        encut_results (Dict): The output of `ENCUT` convergene runs.
        kpspacing_results (Dict): The output of `KSPACING` convergene runs.

    Returns:
        Dict: Final results as a disctionary which can be used for subsequent runs.
    """
    converged_params = {}
    converged_params['ENCUT'] = encut_results.get_dict()['converged_encut']
    converged_params['KSPACING'] = float(kpspacing_results.get_dict()['converged_kspacing'])
    return Dict(dict=converged_params)


class VaspConvergeWorkChain(WorkChain):
    """Convergence WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Expose VaspMultiStageWorkChain inputs
        spec.expose_inputs(VaspMultiStageWorkChain)

        # Define VaspConvergeWorkChain specific inputs
        spec.input('encut_list', valid_type=List, required=True, help='list of ENCUT for convergence calcs.')
        spec.input('kspacing_list', valid_type=List, required=True, help='list of kspacing for convergence calcs.')
        spec.input(
            'offset',
            valid_type=List,
            default=lambda: List(list=[0, 0, 0]),
            required=False,
            help='Offest for kpoints generation'
        )
        spec.input(
            'threshold',
            valid_type=Float,
            default=lambda: Float(0.001),
            required=True,
            help='Threshold to consider energy per atom converged!'
        )

        # Exit codes
        spec.exit_code(
            840, 'ERROR_UNABLE_TO_SETUP', message='You can converge both kspoints and ksapcing at same time!'
        )

        # Define outline
        spec.outline(
            cls.initialize,
            while_(cls.should_converge_encut)(
                cls.run_encut_converge,
                cls.inspect_encut_converge,
                cls.process_encut_converge,
            ),
            while_(cls.should_converge_kspacing)(
                cls.run_kspacing_converge,
                cls.inspect_kspacing_converge,
                cls.process_kspacing_converge,
            ),
            cls.results,
        )
        # Expose outputs
        spec.expose_outputs(VaspMultiStageWorkChain)
        spec.output_namespace('convergence_results', valid_type=Dict, required=False, dynamic=True)

    def initialize(self):
        """Initialize inputs and settings"""
        try:
            self.ctx.encut_list = self.inputs.encut_list
            self.ctx.should_converge_encut = True
        except ValueError:
            self.ctx.should_converge_encut = False

        try:
            self.ctx.kspacing_list = self.inputs.kspacing_list
            self.ctx.should_converge_kspacing = True
        except ValueError:
            self.ctx.should_converge_kspacing = False

        self.ctx.offset = self.inputs.offset.get_list()

        self.ctx.encut_idx = 0
        self.ctx.kspacing_idx = 0

        # Setup inputs
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspMultiStageWorkChain))

    def should_converge_encut(self):
        """Return true until whole ENCUT values are submitted!"""
        return self.ctx.encut_idx < len(self.ctx.encut_list)

    def should_converge_kspacing(self):
        """Return true until whole KSPACING values are submitted!"""
        return self.ctx.kspacing_idx < len(self.ctx.kspacing_list)

    def run_encut_converge(self):
        """Submit VaspMultiStageWorkChain with all items in ENCUT list"""
        for encut in self.ctx.encut_list:
            self.ctx.inputs.parameters = update_incar_encut(self.ctx.inputs.parameters, Int(encut))
            self.ctx.inputs['metadata']['label'] = f'ENCUT_{encut}'
            self.ctx.inputs['metadata']['call_link_label'] = f'run_ENCUT_{encut}'
            running = self.submit(VaspMultiStageWorkChain, **self.ctx.inputs)
            self.ctx.encut_idx += 1
            encut_label = f'encut_{encut}'
            self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for ENCUT:<{encut}>')
            self.to_context(**{encut_label: running})

    def inspect_encut_converge(self):
        """Asserts whether all ENCUT calculations are finished ok"""
        for encut in self.ctx.encut_list:
            assert self.ctx[f'encut_{encut}'].is_finished_ok

    def process_encut_converge(self):
        """Process and extract results of ENCUT convergence"""
        all_encut_outputs = {}
        for encut in self.ctx.encut_list:
            all_encut_outputs[f'encut_{encut}'] = self.ctx[f'encut_{encut}'].outputs.output_parameters
        self.out('convergence_results.ENCUT', identify_encut_convergence(self.inputs.threshold, **all_encut_outputs))

    def run_kspacing_converge(self):
        """Submit VaspMultiStageWorkChain with all items in ENCUT list"""
        converged_encut = self.outputs['convergence_results']['ENCUT']['converged_encut']
        self.ctx.inputs.parameters = update_incar_encut(self.ctx.inputs.parameters, Int(converged_encut))
        for kspacing in self.ctx.kspacing_list:
            kpoints = KpointsData()
            kpoints.set_cell_from_structure(self.inputs.structure)
            kpoints.set_kpoints_mesh_from_density(kspacing, offset=self.ctx.offset)
            self.ctx.inputs.vasp_base.vasp.kpoints = kpoints
            self.ctx.inputs['metadata']['label'] = f'KSPACING_{int(kspacing*1000)}'
            self.ctx.inputs['metadata']['call_link_label'] = f'run_KSPACING_{int(kspacing*1000)}'
            running = self.submit(VaspMultiStageWorkChain, **self.ctx.inputs)
            self.ctx.kspacing_idx += 1
            kspacing_label = f'kspacing_{int(kspacing*1000)}'
            self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for KSPACING:<{kspacing}>')
            self.to_context(**{kspacing_label: running})

    def inspect_kspacing_converge(self):
        """Asserts whether all ENCUT calculations are finished ok"""
        for kspacing in self.ctx.kspacing_list:
            assert self.ctx[f'kspacing_{int(kspacing*1000)}'].is_finished_ok

    def process_kspacing_converge(self):
        """Process and extract results of ENCUT convergence"""
        all_kspacing_outputs = {}
        for kspacing in self.ctx.kspacing_list:
            all_kspacing_outputs[f'kspacing_{int(kspacing*1000)}'] = self.ctx[f'kspacing_{int(kspacing*1000)}'
                                                                              ].outputs.output_parameters
        self.out(
            'convergence_results.KSPACING',
            identify_kspacing_convergence(self.inputs.threshold, **all_kspacing_outputs)
        )

    def results(self):
        """Handle results"""
        self.out(
            'output_parameters',
            return_final_results(
                self.outputs['convergence_results']['ENCUT'], self.outputs['convergence_results']['KSPACING']
            )
        )
        self.report('VaspConvergeWorkChain FInished Successfully!')


# EOF
