"""``BaseWorkChain`` to run a VASP calculation"""
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Oszicar

from aiida.common import AttributeDict
from aiida.engine import calcfunction
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.orm import Dict, FolderData, CalcJobNode, StructureData
from aiida.plugins import CalculationFactory

from aiida_catmat.calcfunctions import dict_merge
from aiida_catmat.parsers import STDERR_ERRS, STDOUT_ERRS

# StructureData = DataFactory('structure')  # pylint: disable=invalid-name
VaspCalculation = CalculationFactory('vasp.vasp')  # pylint: disable=invalid-name


@calcfunction
def update_incar(incar: Dict, modifications: Dict) -> Dict:
    """Updates the current ``INCAR`` with proposed modifications.

    Args:
        incar (Dict): Current ``INCAR``
        modifications (Dict): Proposed modifications to handle the error.
    Returns:
        Dict: The updated ``INCAR``.
    """
    incar = incar.get_dict()
    modifications = modifications.get_dict()
    dict_merge(incar, modifications)
    return Dict(dict=incar)


@calcfunction
def apply_strain_on_structure(retrived_folder: FolderData) -> StructureData:
    """Applies 0.2 strain on structure

    Args:
        retrived_folder (FolderData): The retrieved folder in the repository that contains ``CONTCAR``

    Returns:
        StructureData: The structure after applying 0.2 strain.
    """
    with retrived_folder.open('CONTCAR') as handler:
        structure = Structure.from_file(handler.name)
    structure.apply_strain(0.2)
    return StructureData(pymatgen_structure=structure)


def get_stdout_errs(calculation: CalcJobNode) -> set:
    """Parses the ``_scheduler-stdout.txt`` and searches for pre-defined error messages.

    Args:
        calculation (CalcJobNode): The calculation `Node`.

    Returns:
        set: A set of found error messages in ``_scheduler-stdout.txt``
    """
    errors = set()
    errors_subset_to_catch = list(STDOUT_ERRS.keys())

    with calculation.outputs.retrieved.open('_scheduler-stdout.txt') as handler:
        for line in handler:
            l = line.strip()  #pylint: disable=invalid-name
            for err, msgs in STDOUT_ERRS.items():
                if err in errors_subset_to_catch:
                    for msg in msgs:
                        if l.find(msg) != -1:
                            errors.add(err)
    return errors


def get_stderr_errs(calculation: CalcJobNode) -> set:
    """Parses the ``_scheduler-stderr.txt`` and searches for pre-defined error messages.

    Args:
        calculation (CalcJobNode): The calculation ``Node``.

    Returns:
        set: A set of found error messages in ``_scheduler-stderr.txt``
    """
    errors = set()
    errors_subset_to_catch = list(STDERR_ERRS.keys())

    with calculation.outputs.retrieved.open('_scheduler-stderr.txt') as handler:
        for line in handler:
            l = line.strip()  #pylint: disable=invalid-name
            for err, msgs in STDOUT_ERRS.items():
                if err in errors_subset_to_catch:
                    for msg in msgs:
                        if l.find(msg) != -1:
                            errors.add(err)

    return errors


#pylint: disable=inconsistent-return-statements
#pylint: disable=too-many-public-methods
class VaspBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a ``VASP`` calculation with automated error handling and restarts."""

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
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and then create the
        inputs dictionary in ``self.ctx.inputs``. This ``self.ctx.inputs`` dictionary will be
        used by the ``BaseRestartWorkChain`` to submit the calculations in the internal loop."""

        super().setup()

        self.ctx.inputs = AttributeDict(self.exposed_inputs(VaspCalculation, 'vasp'))
        self.ctx.parameters = self.ctx.inputs.parameters
        self.ctx.modifications = {}
        self.ctx.err_count = {}

    def report_error_handled(self, calculation: CalcJobNode, action: str) -> None:
        """Report an action taken for a calculation that has failed.

        Args:
            calculation (CalcJobNode): the failed calculation node
            action (str): a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))

    @process_handler(priority=570, enabled=True)
    def handle_timeout(self, calculation):
        """Error handler that restarts calculation finished with ``TIMEOUT`` ExitCode."""
        self.ctx.stdout_errors = get_stdout_errs(calculation)
        self.ctx.stderr_errors = get_stderr_errs(calculation)

        if 'walltime' in self.ctx.stderr_errors:
            self.report_error_handled(
                calculation, 'Timeout handler. Adding remote folder as input to use binary restart.'
            )
            self.ctx.inputs.restart_folder = calculation.outputs.remote_folder
            return ProcessHandlerReport(False)

    @process_handler(priority=1, enabled=True)
    def apply_modifications(self, calculation):
        """Apply all requested modifications"""
        if self.ctx.modifications:
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, Dict(dict=self.ctx.modifications))
            self.ctx.modifications = {}
            self.report('Applied all modifications for {}<{}>'.format(calculation.process_label, calculation.pk))
            return ProcessHandlerReport(False)

    @process_handler(priority=300, enabled=True)
    def handle_lreal(self, calculation):
        """Handle ``ERROR_LREAL_SMALL_SUPERCELL`` exit code"""
        if 'lreal' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'LREAL': False})
            action = 'ERROR_LREAL_SMALL_SUPERCELL: LREAL is set to False'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=310, enabled=True)
    def handle_rsphere(self, calculation):
        """Handle ``ERROR_RSPHERE`` exit code"""
        if 'rsphere' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'LREAL': False})
            action = 'ERROR_RSPHERE: LREAL is set to False'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=320, enabled=True)
    def handle_zbrent(self, calculation):
        """Handle ``ERROR_ZBRENT`` exit code"""
        if ('zbrent' in self.ctx.stdout_errors) and (self.ctx.err_count.get('zbrent', 0)):
            ediff = self.ctx.parameters.get_dict().get('EDIFF', 1e-6) * 0.01
            self.ctx.modifications.update({'EDIFF': ediff})
            self.ctx.err_count.update({'zbrent': 1})
            action = f'ERROR_ZBRENT: EDIFF is decreased to {ediff}'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=100, enabled=False)
    def handle_tetrahedron(self, calculation):
        """Handle ``ERROR_TETRAHEDRON`` exit code"""
        if 'tet' in self.ctx.stdout_errors:
            if 'kspacing' in self.ctx.inputs:
                old_kspacing = self.ctx.inputs.kspacing.value
                new_kspacing = old_kspacing * 0.8
                self.ctx.inputs.kspacing = new_kspacing
                action = f'ERROR_TETRAHEDRON: KSPACING is decreased by 80%: <{old_kspacing}> to <{new_kspacing}>'
                self.report_error_handled(calculation, action)
            else:
                self.ctx.modifications.update({'ISMEAR': 0, 'SIGMA': 0.05})
                action = f'ERROR_TETRAHEDRON: ISMEAR is set to 0 and SIGMA 0.05'
                self.report_error_handled(calculation, action)
                return ProcessHandlerReport(False)

    @process_handler(priority=110, enabled=True)
    def handle_inverse_rotation_matrix(self, calculation):
        """Handle ``ERROR_INVERSE_ROTATION_MATRIX`` exit code"""
        if 'inv_rot_mat' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'SYMPREC': 1e-8})
            action = 'ERROR_INVERSE_ROTATION_MATRIX: SYMPREC is decreased to 1E-08'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=120, enabled=True)
    def handle_subspace_matrix(self, calculation):
        """Handle ``ERROR_SUBSPACEMATRIX`` exit code"""
        if 'subspacematrix' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'LREAL': False})
            action = 'ERROR_SUBSPACEMATRIX: LREAL is set to False'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=130, enabled=True)
    def handle_amin(self, calculation):
        """Handle ``ERROR_AMIN`` exit code"""
        if 'amin' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'AMIN': 0.01})
            action = 'ERROR_AMIN: AMIN is set to 0.01'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=140, enabled=True)
    def handle_pricel(self, calculation):
        """Handle ``ERROR_PRICEL`` exit code"""
        if 'pricel' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ISYM': 0, 'SYMPREC': 1e-8})
            action = 'ERROR_PRICEL: ISYM is set to zero and SYMPREC to 1E-08'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=150, enabled=True)
    def handle_brions(self, calculation):
        """Handle ``ERROR_BRIONS`` exit code"""
        if 'brions' in self.ctx.stdout_errors:
            potim = self.ctx.parameters.get_dict().get('POTIM', 0.5) + 0.1
            self.ctx.modifications.update({'POTIM': potim})
            action = f'ERROR_BRIONS: POTIM is set to <{potim}>'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=160, enabled=True)
    def handle_pssyevx(self, calculation):
        """Handle ``ERROR_PSSYEVX`` exit code"""
        if 'pssyevx' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ALGO': 'Normal'})
            action = 'ERROR_PSSYEVX: ALGO is set to Normal'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=170, enabled=True)
    def handle_eddrmm(self, calculation):
        """Handle ``ERROR_EDDRMM`` exit code"""
        if 'eddrmm' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ISTART': 0, 'ICHARG': 2})
            if self.ctx.parameters['ALGO'] in ['Fast', 'VeryFast']:
                self.ctx.modifications.update({'ALGO': 'Normal'})
                action = 'ERROR_EDDRMM: ALGO is set to Normal, ISTART to 0 and ICHARG to 2'
            else:
                potim = self.ctx.parameters.get_dict().get('POTIM', 0.5) / 2.0
                self.ctx.modifications.update({'POTIM': potim})
                action = f'ERROR_EDDRMM: POTIM is set to <{potim}>, ISTART to 0 and ICHARG to 2'
            self.ctx.inputs.restart_folder = calculation.outputs.remote_folder
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=180, enabled=True)
    def handle_edddav(self, calculation):
        """Handle ``ERROR_EDDDAV`` exit code"""
        if 'edddav' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ALGO': 'All', 'ISTART': 0, 'ICHARG': 2})
            action = 'ERROR_EDDDAV: ALGO is set to Normal, ISTART to 0 and ICHARG to 2'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=190, enabled=True)
    def handle_grad_not_orth(self, calculation):
        """Handle ``ERROR_GRAD_NOT_ORTH`` exit code"""
        if 'grad_not_orth' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ISMEAR': 0, 'SIGMA': 0.05})
            action = f'ERROR_GRAD_NOT_ORTH: ISMEAR is set to zero and SIGMA 0.05'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    # TODO: double check the solution #pylint: disable=fixme
    @process_handler(priority=200, enabled=False)
    def handle_zheev(self, calculation):
        """Handle ``ERROR_ZHEEV`` exit code"""
        if 'zheev' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ALGO': 'Exact'})
            action = 'ERROR_ZHEEV: ALGO is set to Exact'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=210, enabled=True)
    def handle_elf_kpar(self, calculation):
        """Handle ``ERROR_ELF_KPAR`` exit code"""
        if 'elf_kpar' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'KPAR': 1})
            action = 'ERROR_ELF_KPAR: KPAR is set to 1'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    # TODO: double check the solution. #pylint: disable=fixme
    @process_handler(priority=220, enabled=False)
    def handle_rhosyg(self, calculation):
        """Handle ``ERROR_RHOSYG`` exit code"""
        if 'rhosyg' in self.ctx.stdout_errors:
            self.ctx.modifications.update({
                'ISYM': 0,
                'SYMPREC': 1e-4  # ??
            })
            action = 'ERROR_RHOSYG: ISYM is set to 0 and SYMPREC to 1E-04'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    # TODO: double check the solution #pylint: disable=fixme
    @process_handler(priority=230, enabled=False)
    def handle_posmap(self, calculation):
        """Handle ``ERROR_POSMAP`` exit code"""
        if 'posmap' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'SYMPREC': 1e-6})
            action = 'ERROR_POSMAP: SYMPREC is set to 1E-06'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=240, enabled=True)
    def handle_point_group(self, calculation):
        """Handle ``ERROR_POINT_GROUP`` exit code"""
        if 'point_group' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'ISYM': 0})
            action = 'ERROR_POINT_GROUP: ISYM is set to 0!'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

    @process_handler(priority=250, enabled=True)
    def handle_zpotrf(self, calculation):
        """Handle ``ERROR_ZPOTRF`` exit code"""
        if 'zportf' in self.ctx.stdout_errors:
            try:
                with calculation.outputs.retrieved.open('OSZICAR') as handler:
                    oszcar = Oszicar(handler.name)
                nsteps = len(oszcar.ionic_steps)
            except Exception:  #pylint: disable=broad-except
                nsteps = 0
            if nsteps >= 0:
                potim = self.ctx.parameters.get_dict().get('POTIM', 0.5) / 2.0
                self.ctx.modifications.update({'ISYM': 0, 'POTIM': potim})
                action = f'ERROR_ZPOTRF: ISYM is set to 0 and POTIM to {potim}!'
            elif self.ctx.parameters.get_dict().get('NSW',
                                                    0) == 0 or self.ctx.parameters.get_dict().get('ISIF',
                                                                                                  0) in range(3):
                self.ctx.modifications.update({'ISYM': 0})
                action = 'ERROR_ZPOTRF: ISYM is set to 0!'
            else:
                self.ctx.inputs.structure = apply_strain_on_structure(calculation.outputs.retrieved)
                action = 'ERROR_ZPOTRF: Applied 0.2 strain on the strcuture from CONTCAR'

            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)


# EOF
