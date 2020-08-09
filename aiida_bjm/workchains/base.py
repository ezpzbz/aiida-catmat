"""Base work chain to run a VASP calculation"""

from aiida.common import AttributeDict
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory


VaspCalculation = CalculationFactory('vasp.vasp')  # pylint: disable=invalid-name
# VaspBaseWorkChain = WorkflowFactory('vasp.vase')

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

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.
        This should be called in a registered error handler if its condition is met and an action was taken.
        :param calculation: the failed calculation node
        :param action: a string message with the action taken"""

        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))

    @process_handler(priority=570, exit_codes=VaspCalculation.exit_codes.TIMEOUT, enabled=True)
    def handle_timeout(self, calculation):
        """Error handler that restarts calculation finished with TIMEOUT ExitCode."""
        self.report_error_handled(calculation, "Timeout handler. Adding remote folder as input to use binary restart.")
        self.ctx.inputs.remote_folder = calculation.outputs.remote_folder
        return ProcessHandlerReport(False)