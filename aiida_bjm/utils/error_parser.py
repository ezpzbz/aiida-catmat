"""Small parser to get the reported
WARNING and ERROR messages in _scheduler-stdout.txt
"""

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
    'lreal': ['Therefore set LREAL=.FALSE. in the  INCAR file']
}

STDERR_ERRS = {'walltime': ['PBS: job killed: walltime'], 'memory': ['job killed: memory']}


def get_stdout_errs(calculation):
    """
    Parses the _scheduler-stdout.txt and reports any found errors.
    """
    errors = set()
    errors_subset_to_catch = list(STDOUT_ERRS.keys())
    error_msgs = set()
    with calculation.outputs.retrieved.open('_scheduler-stdout.txt') as handler:
        for line in handler:
            l = line.strip()  #pylint: disable=invalid-name
            for err, msgs in STDOUT_ERRS.items():
                if err in errors_subset_to_catch:
                    for msg in msgs:
                        if l.find(msg) != -1:
                            errors.add(err)
                            error_msgs.add(msg)
    return errors, error_msgs


def get_stderr_errs(calculation):
    """
    Parses the _scheduler-stderr.txt and reports any found errors.
    """
    errors = set()
    errors_subset_to_catch = list(STDERR_ERRS.keys())
    error_msgs = set()
    with calculation.outputs.retrieved.open('_scheduler-stderr.txt') as handler:
        for line in handler:
            l = line.strip()  #pylint: disable=invalid-name
            for err, msgs in STDERR_ERRS.items():
                if err in errors_subset_to_catch:
                    for msg in msgs:
                        if l.find(msg) != -1:
                            errors.add(err)
                            error_msgs.add(msg)
    return errors, error_msgs
