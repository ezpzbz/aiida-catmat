"""Customized parsers for `aiida-catmat`"""

from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.core import Spin
from parsevasp.outcar import Outcar

from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict, StructureData
# from aiida.plugins import DataFactory

# StructureData = DataFactory('structure')  # pylint: disable=invalid-name
STDOUT_ERRS = {
    'tet': [
        'Tetrahedron method fails for NKPT<4', 'Fatal error detecting k-mesh', 'Fatal error: unable to match k-point',
        'Routine TETIRR needs special values', 'Tetrahedron method fails (number of k-points < 4)'
    ],
    'inv_rot_mat': ['inverse of rotation matrix was not found (increase SYMPREC)'],
    'brmix': ['BRMIX: very serious problems'],
    'subspacematrix': ['WARNING: Sub-Space-Matrix is not hermitian in DAV'],
    'tetirr': ['Routine TETIRR needs special values'],
    'incorrect_shift': ['Could not get correct shifts'],
    'real_optlay': ['REAL_OPTLAY: internal error', 'REAL_OPT: internal ERROR'],
    'rspher': ['ERROR RSPHER', 'RSPHER: internal ERROR:'],
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
    'rsphere': ['RSPHER: internal ERROR'],
    'posmap': ['POSMAP internal error: symmetry equivalent atom not found'],
    'point_group': ['Error: point group operation missing'],
    'aliasing': ['WARNING: small aliasing (wrap around) errors must be expected'],
    'aliasing_incar': ['Your FFT grids (NGX,NGY,NGZ) are not sufficient for an accurate'],
    'lreal': ['Therefore set LREAL=.FALSE. in the  INCAR file'],
}

STDERR_ERRS = {'walltime': ['PBS: job killed: walltime'], 'memory': ['job killed: memory']}
# From https://www.vasp.at/wiki/index.php/GGA
GGA_FUNCTIONALS = {
    '91': 'PW91(Perdew-Wang91)',
    'PE': 'PBE(Perdew-Burke-Ernzerhof)',
    'AM': 'AM05',
    'HL': 'HL(Hendin-Lundqvist)',
    'CA': 'CA(Ceperley-Alder)',
    'PZ': 'Ceperley-Alder, parametrization of Perdew-Zunger',
    'WI': 'Wigner',
    'RP': 'revised Perdew-Burke-Ernzerhof (RPBE) with Pade Approximation',
    'RE': 'revPBE',
    'VW': 'Vosko-Wilk-Nusair (VWN)',
    'B3': 'B3LYP, where LDA part is with VWN3-correlation',
    'B5': 'B3LYP, where LDA part is with VWN5-correlation',
    'BF': 'BEEF, xc (with libbeef)',
    'CO': 'no exchange-correlation',
    'PS': 'Perdew-Burke-Ernzerhof revised for solids (PBEsol)',
    'OR': 'optPBE',
    'BO': 'optB88',
    'MK': 'optB86b',
    'RA': 'new RPA Perdew Wang',
    '03': 'range-separated ACFDT (LDA - sr RPA) mu=0.3A^3',
    '05': 'range-separated ACFDT (LDA - sr RPA) mu=0.5A^3',
    '10': 'range-separated ACFDT (LDA - sr RPA) mu=1.0A^3',
    '20': 'range-separated ACFDT (LDA - sr RPA) mu=2.03A^3',
    'PL': 'new RPA+ Perdew Wang'
}

METAGGA_FUNCTIONALS = {
    'TPSS': 'TPSS',
    'RTPSS': 'revised TPSS',
    'M06L': 'M06-L',
    'MS0': 'MS0',
    'MS1': 'MS1',
    'MS2': 'MS2',
    'MBJ': 'modified Becke-Johnson exchange potential in combination with L(S)DA-correlation',
    'SCAN': 'Strongly constrained and appropriately normed semilocal density functional'
}


class VaspBaseParser(Parser):
    """Basic Parser for VaspCalculation"""

    # pylint: disable=inconsistent-return-statements
    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""

        try:
            _ = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        try:
            with self.retrieved.open('vasprun.xml') as handler:
                vrun = Vasprun(handler.name, parse_eigen=False, parse_potcar_file=False, parse_projected_eigen=False)
        except:  #pylint: disable=bare-except
            vrun = None

        try:
            with self.retrieved.open('OUTCAR') as handler:
                vout = Outcar(handler.name)
        except:  #pylint: disable=bare-except
            vout = None

        errors = self._parse_stdout()

        results, structure = self._parse_results(vrun=vrun, vout=vout, errors=errors)

        self.out('misc', Dict(dict=results))

        if structure:
            if 'converged_magmoms' in results:
                structure.add_spin_by_site(results['converged_magmoms'])
                self.out('structure', StructureData(pymatgen_structure=structure))
            else:
                self.out('structure', StructureData(pymatgen_structure=structure))

    def _parse_stdout(self):
        """
        Parses the _scheduler-stdout.txt and reports any found errors.
        """
        errors = {}
        errors_subset_to_catch = list(STDOUT_ERRS.keys())

        with self.retrieved.open('_scheduler-stdout.txt') as handler:
            for line in handler:
                l = line.strip()  #pylint: disable=invalid-name
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg

        with self.retrieved.open('_scheduler-stderr.txt') as handler:
            for line in handler:
                l = line.strip()  #pylint: disable=invalid-name
                for err, msgs in STDOUT_ERRS.items():
                    if err in errors_subset_to_catch:
                        for msg in msgs:
                            if l.find(msg) != -1:
                                errors[err] = msg

        return errors

    @staticmethod
    def _parse_results(vrun, vout, errors):  #pylint: disable=too-many-statements
        """Parse results"""

        def _site_magnetization(structure, magnetizations):
            site_mags = []
            magmoms = []
            symbols = [specie.symbol for specie in structure.species]
            for symbol, magnetization in zip(symbols, magnetizations):
                mag_dict = {}
                mag_dict[symbol] = magnetization
                magmoms.append(magnetization['tot'])
                magmoms = [0 if abs(mag) < 0.6 else mag for mag in magmoms]
                site_mags.append(mag_dict)
            return site_mags, magmoms

        results = {}
        results['converged'] = False
        results['converged_ionically'] = False
        results['converged_electronically'] = False
        structure = None
        if vrun:
            results['energy_unit'] = 'eV'
            results['band_gap_unit'] = 'eV'
            results['extra_parameters'] = {}
            results['converged'] = vrun.converged
            results['converged_ionically'] = vrun.converged_ionic
            results['converged_electronically'] = vrun.converged_electronic
            results['DFT+U'] = vrun.is_hubbard
            results['potcar_specs'] = vrun.potcar_spec
            results['extra_parameters']['number_of_bands'] = vrun.parameters['NBANDS']
            results['extra_parameters']['number_of_electrons'] = vrun.parameters['NELECT']
            results['extra_parameters']['ebreak'] = vrun.parameters['EBREAK']
            results['extra_parameters']['amix'] = vrun.parameters['AMIX']
            results['extra_parameters']['bmix'] = vrun.parameters['BMIX']
            results['extra_parameters']['amin'] = vrun.parameters['AMIN']
            results['extra_parameters']['amix_mag'] = vrun.parameters['AMIX_MAG']
            results['extra_parameters']['bmix_mag'] = vrun.parameters['BMIX_MAG']
            results['extra_parameters']['imix'] = vrun.parameters['IMIX']
            results['extra_parameters']['ngx'] = vrun.parameters['NGX']
            results['extra_parameters']['ngy'] = vrun.parameters['NGY']
            results['extra_parameters']['ngz'] = vrun.parameters['NGZ']
            results['extra_parameters']['ngxf'] = vrun.parameters['NGXF']
            results['extra_parameters']['ngyf'] = vrun.parameters['NGYF']
            results['extra_parameters']['ngzf'] = vrun.parameters['NGZF']
            results['run_type'] = vrun.run_type
            results['final_energy'] = vrun.final_energy
            results['final_energy_per_atom'] = vrun.as_dict()['output']['final_energy_per_atom']
            results['fermi_energy'] = vrun.efermi
            if vrun.parameters['ISPIN'] == 2:
                results['spin_polarized'] = True
                results['band_gap_spin_up'] = vrun.complete_dos.get_gap(spin=Spin.up)
                results['band_gap_spin_down'] = vrun.complete_dos.get_gap(spin=Spin.down)
                mags = vout.get_magnetization()
                results['total_magnetization'] = mags['full_cell'][0]
            else:
                results['spin_polarized'] = False
                results['band_gap_spin_up'] = vrun.complete_dos.get_gap()
                results['band_gap_spin_down'] = vrun.complete_dos.get_gap()
            results['errors'] = errors
            if vrun.incar['NSW'] != 0:
                structure = vrun.final_structure
            # if 'LORBIT' in vrun.incar:
            if vrun.incar.get('LORBIT', None) > 10 and vrun.parameters['ISPIN'] == 2:
                magns = _site_magnetization(vrun.final_structure, list(mags['sphere']['x']['site_moment'].values()))
                results['complete_site_magnetizations'] = magns[0]
                results['converged_magmoms'] = magns[1]
        else:
            results['errors'] = errors

        return results, structure


# EOF
