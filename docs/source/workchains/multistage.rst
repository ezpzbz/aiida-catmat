=====================================
VaspMultiStageWorkChain
=====================================
The inspiration to write the ``VaspMultiStageWorkChain`` comes from 
the ``Cp2kMultiStageWorkChain`` of `aiida-lsmo <https://github.com/lsmo-epfl/aiida-lsmo>`_ package.
In a nutshell, DFT codes are caple of performing different types of calculations. For instance, we
can relax atomic position, lattice parameters, and/or unit cell shape. Consequently, we can use the 
resulting structure to perform more accurate electronic structure calculations. The aim of multistage approach
would be bringing all different calculations which we want to perform under a same umberalla. Therfore, we can
make an automated, robus, and reproducible pipeline of calculations which not necessirly would be done using 
same DFT code or same calculation setup. This important feature is enabled by providing the combinations/settings in 
the form of ``protocols``.

Protocols and how they work
----------------------------

The :py:func:`~aiida_catmat.workchains.vasp_multistage.VaspMultiStageWorkChain` allows us to combine any sequence of
``VASP`` calculation. This sequence of calculations can be provided as a ``YAML`` file. Although I provide a serties of 
different protocols, these can be easily supplied by user too. The shipped protocols with ``aiida-catmat`` are 
available in ``workchains/protocols/vasp``. In the name of protocols, ``R`` and ``S`` stand for ``Relaxation`` and 
``Static`` calculations. The ``03`` and ``3`` both mean using ``ISIF=3`` in relaxation, however, the ``03`` only
runs for 5 steps. The sequences of letters and numbers, shows the sequence of stages. For instance, ``R03R3S`` is 
a protocol that runs a short 5-step relaxation (``R03``), then inspects and tries to modify ``INCAR`` if it is ncessary, then
runs a full relaxation (``R3``), and finally runs a static calculation (``S``) with slightly increased ``ENCUT`` and
decreased ``KSPACING`` to obtain more acurate energies. The protocol file looks like::

    stage_0:
        ALGO: Normal
        EDIFF: 1.0e-06
        ENCUT: 650
        IBRION: 2
        ISIF: 3
        ISMEAR: 0
        ISPIN: 2
        LORBIT: 11
        LREAL: Auto
        LWAVE: true
        NELM: 200
        NSW: 5
        PREC: Accurate
        SIGMA: 0.05
    stage_1:
        ALGO: Normal
        EDIFF: 1.0e-06
        EDIFFG: -0.01
        ENCUT: 650
        IBRION: 2
        ISIF: 3
        ISMEAR: 0
        ISPIN: 2
        LORBIT: 11
        LREAL: Auto
        LWAVE: true
        NELM: 200
        NSW: 400
        PREC: Accurate
        SIGMA: 0.05
    stage_2:
        ALGO: Normal
        EDIFF: 1.0e-07
        ENCUT: 700
        IBRION: -1
        ISIF: 3
        ISMEAR: 0
        ISPIN: 2
        LAECHG: true
        LCHARG: true
        LORBIT: 11
        LREAL: false
        LVHAR: true
        LWAVE: false
        NELM: 200
        NSW: 0
        PREC: Accurate
        SIGMA: 0.05

Currently, users can supply their own protocol either by modifying the settings in one of the exisiting 
protocols.
To provide the user settings, one can define a dictionary and pass it as parameters::

    user_incar = {
        'NPAR': 1,
        'GGA': 'PS',
        'ISPIN': 2,
        'ENCUT': 500,
        'LDAU': False,
    }
    builder = VaspMultiStageWorkChain.get_builder()
    builder.parameters = Dict(dict=user_incar)

This will either add extra tags to the load protocol or overwrites the exisiting ones.

Kpoints
-------
There are two ways to supply ``kpoints`` to the workchain. We can use the ``kpoints`` input::

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])
    builder.vasp_base.vasp.kpoints = kpoints

or we can provide the ``kspacing`` tag where we can also make the gamma-centered or force parity in
the generated mesh::

    builder.kspacing = Float(0.242)
    builder.kgamma = Bool(True)
    builder.force_parity = Bool(True)

POTCAR sets
-----------
Currently, there are two sets of ``POTCAR`` mappings available:

1. ``MPRelaxSet``: these are ones taken from the Materials Project relax set.
2. ``VASP``: these are ones recommended by `VASP <https://www.vasp.at/wiki/index.php/Available_PAW_potentials>`_


Hubbard parameters
------------------
Similar to ``POTCAR`` descriptions, there are two sets of ``U`` parameters available which both are 
extracted from Materials Project datasets:

1. ``MITSet``
2. ``MPSet``


Output dictionary
------------------
Once the calculation is finished, we will have a dictionary which containes really loads of parsed information
for each stage of calculation from ``vasprun.xml`` and ``OUTCAR`` which looks like::

    "stage_0_static": {
        "DFT+U": false,
        "band_gap_spin_down": 0.0,
        "band_gap_spin_up": 0.0,
        "band_gap_unit": "eV",
        "converged": true,
        "converged_electronically": true,
        "converged_ionically": true,
        "converged_magmoms": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        "energy_unit": "eV",
        "errors": {},
        "extra_parameters": {
            "amin": 0.1,
            "amix": 0.4,
            "amix_mag": 1.6,
            "bmix": 1.0,
            "bmix_mag": 1.0,
            "ebreak": 1.32e-06,
            "imix": 4,
            "ngx": 36,
            "ngxf": 72,
            "ngy": 36,
            "ngyf": 72,
            "ngz": 36,
            "ngzf": 72,
            "number_of_bands": 19,
            "number_of_electrons": 24.0
        },
        "fermi_energy": 0.49552331,
        "final_energy": -15.67042079,
        "final_energy_per_atom": -1.95880259875,
        "potcar_specs": [
            {
                "hash": null,
                "titel": "PAW_PBE Li_sv 10Sep2004"
            }
        ],
        "run_type": "PBEsol",
        "spin_polarized": true,
        "total_magnetization": -0.3907086

Moreover, if the relaxation is invloved, we would have relaxed structure as an output of the workchain. The fina 
``INCAR`` for each stage of workchain also is reported::

    Outputs                 PK    Type
    ----------------------  ----  -------------
    final_incar
        stage_0_static      792   Dict
        stage_1_relaxation  802   Dict
        stage_2_static      813   Dict
    output_parameters       821   Dict
    structure               809   StructureData



Detailed inputs, outputs, and outline
-------------------------------------
.. aiida-workchain:: VaspMultiStageWorkChain
    :module: aiida_catmat.workchains

