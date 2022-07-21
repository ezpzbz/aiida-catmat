"""VaspMultiStageDdecWorkChain
It wraps VaspMultiStageWorkChain to a single point calculation and
consecuently calculates the atomic charges and spin moments using DDEC method.
"""
# from aiida.orm import CifData
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext
from aiida.plugins import CalculationFactory, WorkflowFactory

VaspMultiStageWorkChain = WorkflowFactory('catmat.vasp_multistage')  #pylint: disable=invalid-name
DdecCalculation = CalculationFactory('ddec')  # pylint: disable=invalid-name
# CifData = DataFactory('cif')  # pylint: disable=invalid-name


def get_remote_folder(wcnode):
    """Get remote folder from worchain"""
    calcjobs = []
    descendants = wcnode.called_descendants
    for desc in descendants:
        if desc.process_label == 'VaspCalculation':
            calcjobs.append(desc)
    cj_sorted = [(c.ctime, c) for c in calcjobs]
    cj_sorted.sort(key=lambda x: x[0])
    return cj_sorted[-1][1].outputs.remote_folder


class VaspMultiStageDdecWorkChain(WorkChain):
    """VASP+DDEC WorkChain"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # Expose inputs
        spec.expose_inputs(VaspMultiStageWorkChain)
        spec.expose_inputs(DdecCalculation, namespace='ddec')

        # Define outline
        spec.outline(
            cls.initialize,
            cls.run_vasp,
            cls.run_ddec,
            cls.results,
        )
        # Expose outputs
        spec.expose_outputs(DdecCalculation, include=['structure_ddec', 'structure_ddec_spin'])

    def initialize(self):
        """Initialize inputs and settings"""
        # Setup inputs
        self.ctx.vasp_inputs = AttributeDict(self.exposed_inputs(VaspMultiStageWorkChain))
        self.ctx.ddec_inputs = AttributeDict(self.exposed_inputs(DdecCalculation, 'ddec'))

    def run_vasp(self):
        """Submit VaspMultiStageWorkChain"""
        self.ctx.vasp_inputs['metadata']['label'] = 'vasp_multistage'
        self.ctx.vasp_inputs['metadata']['call_link_label'] = 'run_vasp_multistage'
        running = self.submit(VaspMultiStageWorkChain, **self.ctx.vasp_inputs)
        self.report(f'Submitted VaspMultiStageWorkChain <pk>:{running.pk}!')
        return ToContext(vasp=running)

    def run_ddec(self):
        """Submit DdecCalculation"""
        self.ctx.ddec_inputs['charge_density_folder'] = get_remote_folder(self.ctx.vasp)
        self.ctx.ddec_inputs['metadata']['label'] = 'ddec_calculation'
        self.ctx.ddec_inputs['metadata']['call_link_label'] = 'run_ddec_calculation'
        running = self.submit(DdecCalculation, **self.ctx.ddec_inputs)
        self.report(f'Submitted DDEC calculation <pk>:{running.pk}!')
        return ToContext(ddec_calc=running)

    def results(self):
        """Handle results"""
        self.out_many(self.exposed_outputs(self.ctx.vasp, VaspMultiStageWorkChain))
        self.out_many(self.exposed_outputs(self.ctx.ddec_calc, DdecCalculation))
        self.report('VaspMultiStageDdecWorkChain is successfully finished!')


# EOF
