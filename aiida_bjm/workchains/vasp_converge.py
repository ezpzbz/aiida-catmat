"""VaspConvergeWorkChain
It wraps VaspMultiStageWorkChain to series of static calculations for
getting converged ENCUT and KPOINT mesh
"""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import calcfunction, WorkChain, while_
from aiida.plugins import DataFactory, WorkflowFactory

VaspMultiStageWorkChain = WorkflowFactory('bjm.vasp_multistage')  #pylint: disable=invalid-name
PotcarData = DataFactory('vasp.potcar')  #pylint: disable=invalid-name
KpointsData = DataFactory('array.kpoints')  #pylint: disable=invalid-name


@calcfunction
def update_incar_encut(parameters, encut):
    param = parameters.get_dict()
    param.update({'ENCUT': encut.value})
    return orm.Dict(dict=param)


@calcfunction
def identify_convergence(threshold, **all_outputs):
    """Identify converged params"""
    results = {}
    results['final_energy'] = {}
    results['final_energy_per_atom'] = {}

    for key, value in all_outputs.items():
        encut = key.split('_')[1]
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
    return orm.Dict(dict=results)


class VaspConvergeWorkChain(WorkChain):
    """Convergence WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Expose VaspMultiStageWorkChain inputs
        spec.expose_inputs(VaspMultiStageWorkChain)

        # Define VaspConvergeWorkChain specific inputs
        spec.input('encut_list', valid_type=orm.List, required=False, help='list of ENCUT for convergence calcs.')
        spec.input(
            'kpoints_list', valid_type=orm.List, required=False, help='list of kpoints mesh for convergence calcs.'
        )
        spec.input('kspacing_list', valid_type=orm.List, required=False, help='list of kspacing for convergence calcs.')
        spec.input(
            'threshold',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.001),
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
            ),
            # while_(cls.should_converge_kmesh)(
            #     cls.run_kmesh_converge,
            #     cls.inspect_kmesh_converge,
            # ),
            cls.results,
        )
        # Expose outputs
        spec.expose_outputs(VaspMultiStageWorkChain)
        spec.output_namespace('convergence_results', valid_type=orm.Dict, required=False, dynamic=True)

    def initialize(self):
        """Initialize inputs and settings"""
        try:
            self.ctx.encut_list = self.inputs.encut_list
            self.ctx.should_converge_encut = True
        except ValueError:
            self.ctx.should_converge_encut = False
        self.ctx.encut_idx = 0
        # try:
        #     self.ctx.kpoints_list = self.inputs.kpoints_list
        #     self.ctx.should_converge_kpoints = True
        # except ValueError:
        #     self.ctx.should_converge_kpoints = False

        # try:
        #     self.ctx.kspacing_list = self.inputs.kspacing_list
        #     self.ctx.should_converge_kspacing = True
        # except ValueError:
        #     self.ctx.should_converge_kspacing = False

        # if self.ctx.should_converge_kpoints and self.ctx.should_converge_kspacing:
        #     return self.exit_codes.ERROR_UNABLE_TO_SETUP

        # Setup inputs
        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspMultiStageWorkChain))
        # self.ctx.inputs.protocol_tag = orm.Str('S_test')

    def should_converge_encut(self):
        """ENCUT"""
        return self.ctx.encut_idx < len(self.ctx.encut_list)
        # return self.ctx.should_converge_encut

    def should_converge_kmesh(self):
        """KMESH"""
        return self.ctx.should_converge_kpoints or self.ctx.should_converge_kspacing

    def run_encut_converge(self):
        """Submit VaspMultiStageWorkChain"""
        for encut in self.ctx.encut_list:
            self.ctx.inputs.parameters = update_incar_encut(self.ctx.inputs.parameters, orm.Int(encut))
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

    def run_kmesh_converge(self):
        """Submit VaspMultiStageWorkChain"""
        for kmesh in self.ctx.kmesh_list:
            # update incar
            inputs = self.ctx.inputs
            running = self.submit(VaspMultiStageWorkChain, **inputs)
            kmesh_label = f'kmesh_{kmesh}'
            self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk} for KMESH:<{kmesh}>')
            self.to_context(**{kmesh_label: running})

    def inspect_kmesh_converge(self):
        """Asserts whether all ENCUT calculations are finished ok"""
        for kmesh in self.ctx.kmesh_list:
            assert self.ctx[f'kmesh_{kmesh}'].is_finished_ok

    def results(self):
        """Handle results"""
        all_outputs = {}
        for encut in self.ctx.encut_list:
            all_outputs[f'encut_{encut}'] = self.ctx[f'encut_{encut}'].outputs.output_parameters
        self.report(all_outputs)
        self.out('convergence_results.ENCUT', identify_convergence(self.inputs.threshold, **all_outputs))
        self.out('output_parameters', identify_convergence(self.inputs.threshold, **all_outputs))
        # self.out_many(self.exposed_outputs(self.ctx[f'encu_{encut}'], VaspMultiStageWorkChain))
        self.report('VaspConvergeWorkChain FInished Successfully!')


# EOF
