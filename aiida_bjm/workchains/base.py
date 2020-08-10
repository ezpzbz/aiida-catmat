"""Base work chain to run a VASP calculation"""

from aiida.common import AttributeDict
from aiida.engine import calcfunction
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.orm import Dict
from aiida.plugins import CalculationFactory
from aiida_bjm.calcfunctions import dict_merge
from aiida_bjm.utils import get_stdout_errs, get_stderr_errs

VaspCalculation = CalculationFactory('vasp.vasp')  # pylint: disable=invalid-name

@calcfunction
def update_incar(incar, modifications):
    """Merge two aiida Dict objects."""
    incar = incar.get_dict()

    if isinstance(modifications, Dict):
        modifications = modifications.get_dict()

    dict_merge(incar, modifications)

    return Dict(dict=incar)

class VaspBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a VASP calculation with automated error handling and restarts."""

    _process_class = VaspCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(VaspCalculation, namespace='vasp')
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.expose_outputs(VaspCalculation)

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.
        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop."""

        super().setup()

        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspCalculation, 'vasp'))
        self.ctx.parameters = self.ctx.inputs.parameters
        
    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.
        This should be called in a registered error handler if its condition is met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken"""

        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))
    
    @process_handler(priority=570, enabled=True)
    def handle_timeout(self, calculation):
        """Error handler that restarts calculation finished with TIMEOUT ExitCode."""
        self.ctx.stdout_errors = get_stdout_errs(calculation)
        self.ctx.stderr_errors = get_stderr_errs(calculation)

        if 'walltime' in self.ctx.stderr_errors[0]:
            self.report_error_handled(calculation, "Timeout handler. Adding remote folder as input to use binary restart.")
            self.ctx.inputs.restart_folder = calculation.outputs.remote_folder
            return ProcessHandlerReport(False)
    
    @process_handler(priority=100, enabled=True)
    def handle_tetrahedron(self, calculation):
        """Handle '' exit code"""
        if 'tet' in self.ctx.stdout_errors[0]:
            if 'KSPACING' in self.ctx.parameters.get_dict():
                old_kspacing = self.ctx.parameters['KSPACING']
                new_kspacing = self.ctx.parameters['KSPACING'] * 0.8
                modifications = Dict(dict={
                    'KSPACING': new_kspacing
                })
                self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
                action = f'KSPACING decreased by 80%: <{old_kspacing}> to <{new_kspacing}>'
                self.report_error_handled(calculation, action)
            else:
                modifications = Dict(dict={
                    'ISMEAR': 0,
                    'SIGMA': 0.05
                })
                self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
                action = f'Changed to Gaussian smearing with sigma value of 0.05'
                self.report_error_handled(calculation, action)
            
                return ProcessHandlerReport(False)
        
    @process_handler(priority=110)
    def handle_inverse_rotation_matrix(self, calculation):
        """Handle 'ERROR_INVERSE_ROTATION_MATRIX' exit code"""
        if 'inv_rot_mat' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'SYMPREC': 1e-8
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Decreased SYMPREC to 1E-08'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)
    
    @process_handler(priority=120)
    def handle_subspace_matrix(self, calculation):
        """Handle 'ERROR_SUBSPACEMATRIX' exit code"""
        if 'subspacematrix' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'LREAL': False
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set LREAL to FALSE'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=130)
    def handle_amin(self, calculation):
        """Handle 'ERROR_SUBSPACEMATRIX' exit code"""
        if 'amin' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'AMIN': 0.01
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set AMIN to 0.01'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)
    
    @process_handler(priority=140)
    def handle_pricel(self, calculation):
        """Handle 'ERROR_PRICEL' exit code"""
        if 'pricel' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ISYM': 0,
                'SYMPREC': 1e-8
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set ISYM to Zero and Decreased SYMPREC to 1E-08'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=150)
    def handle_brions(self, calculation):
        """Handle 'ERROR_BRIONS' exit code"""
        if 'brions' in self.ctx.stdout_errors[0]:
            potim = self.ctx.parameters.get_dict().get('POTIM', 0.5) + 0.1
            modifications = Dict(dict={
                'POTIM': potim
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = f'Set POTIM to <{potim}>'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)
    
    @process_handler(priority=160)
    def handle_pssyevx(self, calculation):
        """Handle 'ERROR_PSSYEVX' exit code"""
        if 'pssyevx' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ALGO': 'Normal'
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set ALGO to Normal'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)
    
    @process_handler(priority=170)
    def handle_eddrmm(self, calculation):
        """Handle 'ERROR_EDDRMM' exit code
        TODO: CHGCAR and WAVECAR have to be deleted or not copied.
        """
        if 'eddrmm' in self.ctx.stdout_errors[0]:
            if self.ctx.parameters['ALGO'] in ['Fast', 'VeryFast']:
                modifications = Dict(dict={
                    'ALGO': 'Normal'
                })
                action = 'Set ALGO to Normal'
            else:
                potim = self.ctx.parameters.get_dict().get('POTIM', 0.5) / 2.0
                modifications = Dict(dict={
                    'POTIM': potim
                })
                action = f'Set POTIM to <{potim}>'
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=180)
    def handle_edddav(self, calculation):
        """Handle 'ERROR_EDDDAV' exit code
        TODO: CHGCAR ahas to be deleted or not copied.
        """
        if 'edddav' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ALGO': 'Normal'
            })
            action = 'Set ALGO to All'
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False) 

    @process_handler(priority=190)
    def handle_grad_not_orth(self, calculation):
        """Handle 'ERROR_GRAD_NOT_ORTH' exit code"""
        if 'grad_not_orth' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ISMEAR': 0,
                'SIGMA': 0.05
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = f'Changed to Gaussian smearing with sigma value of 0.05'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=200)
    def handle_zheev(self, calculation):
        """Handle 'ERROR_ZHEEV' exit code"""
        if 'zheev' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ALGO': 'Exact'
            })
            action = 'Set ALGO to Exact'
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=210)
    def handle_elf_kpar(self, calculation):
        """Handle 'ERROR_ELF_KPAR' exit code"""
        if 'elf_kpar' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'KPAR': 1
            })
            action = 'Set KPAR to 1'
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False) 

    @process_handler(priority=220)
    def handle_rhosyg(self, calculation):
        """Handle 'ERROR_RHOSYG' exit code"""
        if 'rhosyg' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ISYM': 0,
                'SYMPREC': 1e-4 # ??
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set ISYM to Zero and Decreased SYMPREC to 1E-04'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=230)
    def handle_posmap(self, calculation):
        """Handle 'ERROR_POSMAP' exit code"""
        if 'posmap' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'SYMPREC': 1e-6
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set SYMPREC to 1E-06'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)
    
    @process_handler(priority=240)
    def handle_point_group(self, calculation):
        """Handle 'ERROR_POINT_GROUP' exit code"""
        if 'point_group' in self.ctx.stdout_errors[0]:
            modifications = Dict(dict={
                'ISYM': 0
            })
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, modifications)
            action = 'Set ISYM to Zero!'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)