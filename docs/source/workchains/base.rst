=====================================
VaspBaseWorkChain
=====================================
The ``VaspBaseWorkChain`` plays an essntial role in making ``VaspCalculation`` robust and
reproducible. It subclasses ``BaseRestartWorkChain`` from ``aiida-core`` to give us the 
power in inspecting, fixing, and restarting the ``VaspCalculation``. 

The :py:func:`~aiida_catmat.workchains.base.VaspBaseWorkChain` allows us to benefit from `error handlers` to 
spot errors, and provide possible solutions to mitigate them, and restart the calculation after applying necessary modifications.

How error handlers work
+++++++++++++++++++++++
``VaspBaseWorkChain`` parses the `standard output` (``_scheduler-stdout.txt``) and `standard error` (``_scheduler-stdout.txt``)
files of the calculation and reports any pre-defined error messages. These are assigned to 
``self.ctx.stdout_errors`` and ``self.ctx.stderr_errors``. Then, we have separate functions which check for the
existence of different error messages in the ``self.ctx.stdout_errors`` and ``self.ctx.stderr_errors`` and take
appropriate action to mitigate them, if they exist. For example, let's look at the ``handle_lreal`` handler::

    @process_handler(priority=300, enabled=True)
    def handle_lreal(self, calculation):
        """Handle ``ERROR_LREAL_SMALL_SUPERCELL`` exit code"""
        if 'lreal' in self.ctx.stdout_errors:
            self.ctx.modifications.update({'LREAL': False})
            action = 'ERROR_LREAL_SMALL_SUPERCELL: LREAL is set to False'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(False)

If the supercell size would be small, ``VASP`` suggests to set ``LREAL = False``. The ``handle_lreal`` handler
is notified about this by the existence of ``lreal`` key in ``self.ctx.stdout_errors`` and updates a ``self.ctx.modifications``
dictionary which is constructured at the beginning of inspection and is updated continusly. 
Once all checkes are done, the ``apply_modifications`` handler::

    @process_handler(priority=1, enabled=True)
    def apply_modifications(self, calculation):
        """Apply all requested modifications"""
        if self.ctx.modifications:
            self.ctx.inputs.parameters = update_incar(self.ctx.parameters, Dict(dict=self.ctx.modifications))
            self.ctx.modifications = {}
            self.report('Applied all modifications for {}<{}>'.format(calculation.process_label, calculation.pk))
            return ProcessHandlerReport(False)

will update the ``INCAR`` with these modifications and makes it ready for the subsequent run.

The ``process_handler`` decorator enables us to assign different priorities to the handles and also enable/disable them.

Current error handlers
+++++++++++++++++++++++
The current error handlers are taken from the `custodian <https://github.com/materialsproject/custodian>`_ package and translated to the 
``AiiDA`` compatibale methods.


How to add new ``error handler``
++++++++++++++++++++++++++++++++
We may come up with new situations where a new error handler is needed.
If the error message already is defined in the package and only the action needs to be modified,
it should be strightforward by finding the relevant handler in :py:func:`~aiida_catmat.workchains.base.VaspBaseWorkChain` and
updating it.
However, if a new error message is reported by ``VASP``, it first needs to be added to 
:py:func:`~aiida_catmat.parsers` with a unique key. Then, a new handler can be added to the 
:py:func:`~aiida_catmat.workchains.base.VaspBaseWorkChain`. 


Detailed inputs, outputs, and outline
+++++++++++++++++++++++++++++++++++++
.. aiida-workchain:: VaspBaseWorkChain
    :module: aiida_catmat.workchains