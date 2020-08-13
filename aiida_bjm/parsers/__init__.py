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
from pymatgen.electronic_structure.core import Spin

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

        try:
            with self.retrieved.open('vasprun.xml') as handler:
                vrun = Vasprun(handler.name)
            with self.retrieved.open('OUTCAR') as handler:
                vout = Outcar(handler.name)
        except AttributeError:
            vrun = None
            vout = None

        errors = self._parse_stdout()
        
        results, structure = self._parse_results(vrun=vrun, vout=vout, errors=errors)

        self.out('misc', Dict(dict=results))
        
        if structure is not None:
            self.out('structure', StructureData(pymatgen_structure=structure))

    def _parse_stdout(self):
        """
        Parses the _scheduler-stdout.txt and reports any found errors.
        """
        errors = {}
        errors_subset_to_catch = list(STDOUT_ERRS.keys())
        
        with self.retrieved.open('_scheduler-stdout.txt') as handler:
            for line in handler:
                l = line.strip()
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg
        
        with self.retrieved.open('_scheduler-stderr.txt') as handler:
            for line in handler:
                l = line.strip()
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg
        
        return errors
    
    @staticmethod
    def _parse_results(vrun, vout, errors):
        
        def _site_magnetization(structure, magnetizations):
            site_mags = []
            symbols = [specie.symbol for specie in structure.species]
            for symbol, magnetization in zip(symbols, magnetizations):
                mag_dict = {}
                mag_dict[symbol] = magnetization
                site_mags.append(mag_dict)
            return site_mags

        results = {}
        if vrun:
            
            results['converged'] = vrun.converged
            results['converged_ionically'] = vrun.converged_ionic
            results['converged_electronically'] = vrun.converged_electronic
            results['total_energies'] = {}
            results['total_energies']['energy_no_entropy'] = vrun.final_energy
            results['band_gap'] = {}
            if vrun.parameters['ISPIN'] == 2:
                results['band_gap']['spin_up'] = vrun.complete_dos.get_gap(spin=Spin.up)
                results['band_gap']['spin_down'] = vrun.complete_dos.get_gap(spin=Spin.down)
            else:
                results['band_gap']['spin_up'] = vrun.complete_dos.get_gap()
                results['band_gap']['spin_down'] = vrun.complete_dos.get_gap()
            results['errors'] = errors
            results['total_magnetization'] = vout.total_mag
            if 'LORBIT' in vrun.incar:
                results['site_magnetizations'] = _site_magnetization(vrun.final_structure, vout.magnetization)
            if vrun.incar['NSW'] != 0:
                structure = vrun.final_structure
            else:
                structure = None
        else:
            results['errors'] = errors

        return results, structure
