""" AiiDA-BJM Parsers """
import io
import os
from aiida.common import exceptions

from aiida.parsers import Parser
from aiida.common import OutputParsingError, NotExistent
from aiida.engine import ExitCode
from aiida.orm import Dict
from aiida.plugins import DataFactory

from pymatgen.io.vasp import Vasprun, Outcar, Oszicar, Poscar

StructureData = DataFactory('structure')  # pylint: disable=invalid-name
STDOUT_ERRS = {
    'tet': [
        'Tetrahedron method fails for NKPT<4',
        'Fatal error detecting k-mesh',
        'Fatal error: unable to match k-point',
        'Routine TETIRR needs special values',
        'Tetrahedron method fails (number of k-points < 4)'
    ],
    'inv_rot_mat': ['inverse of rotation matrix was not found (increase SYMPREC)'],
    'brmix': ['BRMIX: very serious problems'],
    'subspacematrix': ['WARNING: Sub-Space-Matrix is not hermitian in DAV'],
    'tetirr': ['Routine TETIRR needs special values'],
    'incorrect_shift': ['Could not get correct shifts'],
    'real_optlay': [
        'REAL_OPTLAY: internal error', 
        'REAL_OPT: internal ERROR'
    ],
    'rspher': ['ERROR RSPHER'],
    'dentet': ['DENTET'],
    'too_few_bands': ['TOO FEW BANDS'],
    'triple_product': ['ERROR: the triple product of the basis vectors'],
    'rot_matrix': ['Found some non-integer element in rotation matrix'],
    'brions': ['BRIONS problems: POTIM should be increased'],
    'pricel': ['internal error in subroutine PRICEL'],
    'zpotrf': ['LAPACK: Routine ZPOTRF failed'],
    'amin': ['One of the lattice vectors is very long (>50 A), but AMIN'],
    'zbrent': ['ZBRENT: fatal internal in', 'ZBRENT: fatal error in bracketing'],
    'pssyevx': ['ERROR in subspace rotation PSSYEVX'],
    'eddrmm': ['WARNING in EDDRMM: call to ZHEGV failed'],
    'edddav': ['Error EDDDAV: Call to ZHEGV failed'],
    'grad_not_orth': ['EDWAV: internal error, the gradient is not orthogonal'],
    'nicht_konv': ['ERROR: SBESSELITER : nicht konvergent'],
    'zheev': ['ERROR EDDIAG: Call to routine ZHEEV failed!'],
    'elf_kpar': ['ELF: KPAR>1 not implemented'],
    'elf_ncl': ['WARNING: ELF not implemented for non collinear case'],
    'rhosyg': ['RHOSYG internal error'],
    'posmap': ['POSMAP internal error: symmetry equivalent atom not found'],
    'point_group': ['Error: point group operation missing'],
    'aliasing': ['WARNING: small aliasing (wrap around) errors must be expected'],
    'aliasing_incar': ['Your FFT grids (NGX,NGY,NGZ) are not sufficient for an accurate'],
    }

STDERR_ERRS = {
    'walltime': ['PBS: job killed: walltime'],
    'memory': ['job killed: memory']
}
class VaspBaseParser(Parser):
    """Basic Parser for VaspCalculation"""
    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""

        try:
            _ = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        ALWAYS_RETRIEVE_LIST = [ #pylint: disable=invalid-name

            'CONTCAR', 'OUTCAR', 'vasprun.xml', 'EIGENVAL', 'DOSCAR', '_scheduler-stdout.txt', '_scheduler-stderr.txt'
        ]

        if 'ADDITIONAL_RETRIEVE_LIST' in self.node.inputs.settings.get_dict():
            ALWAYS_RETRIEVE_LIST.append(self.node.inputs.settings.get_dict()['ADDITIONAL_RETRIEVE_LIST'])
        
        for item in ALWAYS_RETRIEVE_LIST:
            if item not in self.retrieved.list_object_names():
                return self.exit_codes.ERROR_CRITICAL_MISSING_FILE
        
        errors = self._parse_stdout()
        results = self._parse_vasprun

        self.out('errors', Dict(dict=errors))
        self.out('output_parameters', Dict(dict=results))
        # self.out('relaxed.structure', StructureData(pymatgen_structure=structure))

    def _parse_stdout(self):
        """
        Parses the _scheduler-stdout.txt and reports any found errors.
        """
        # errors = set()
        errors = {}
        errors_subset_to_catch = list(STDOUT_ERRS.keys())
        # error_msgs = set()
        with self.retrieved.open('_scheduler-stdout.txt') as handler:
            for line in handler:
                l = line.strip()
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg
                                # errors.add(err)
                                # error_msgs.add(msg)
        with self.retrieved.open('_scheduler-stderr.txt') as handler:
            for line in handler:
                l = line.strip()
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg
        
        return errors
    
    def _parse_vasprun(self):
        results = {}
        with self.retrieved.open('vasprun.xml') as handler:
            vasprun = Vasprun(handler.name)
        results['converged'] = vasprun.converged
        results['converged_ionically'] = vasprun.converged_ionic
        results['converged_electronically'] = vasprun.converged_electronic
        results['total_energy'] = vasprun.final_energy

        # structure = vasprun.final_structure

        return results
