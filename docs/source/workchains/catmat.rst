=====================================
VaspCatMatWorkChain
=====================================
The ``VaspCatMatWorkChain`` automates the calculation of cathode materials.


How it works
----------------------------
The :py:func:`~aiida_catmat.workchains.vasp_catmat.VaspCatMatWorkChain` allows us to have a quick way to calculate
cathode properties between two states of charge (ie, fully charged and fully discharged). It is technically able
to work with different working ions such as ``Li`` and ``Na``.
We provide a structure at fully discharged (lithiated) state, it calls the :py:func:`~aiida_catmat.workchains.vasp_multistage.VaspMultiStageWorkChain`
to relax the structure and gets energies, then it removes all ``Li`` ions (for instance), and again calls the 
:py:func:`~aiida_catmat.workchains.vasp_multistage.VaspMultiStageWorkChain`. Once the enrgies and structural information
for both states are available, it calculates relevant properties as reports them as::

    {
        "anode_chemical_potential": -1.9700200833333,
        "battery_type": "Li-ion",
        "change_direction": "charging",
        "energy_of_charged_state": -523.27109795,
        "energy_of_discharged_state": -602.70270521,
        "energy_unit": "eV",
        "gravimetric_energy_density": 363.0,
        "gravimetric_energy_density_unit": "Wh/kg",
        "gravimetric_specific_capacity": 121.2,
        "gravimetric_specific_capacity_unit": "mAh/g",
        "lattice_a_change": 0.041,
        "lattice_b_change": 0.041,
        "lattice_c_change": -14.737,
        "lattice_parameters_unit": "A",
        "number_of_extracted_Li_ions": 16,
        "open_circuit_voltage": 2.99,
        "open_circuit_voltage_unit": "V",
        "volume_change": -14.7,
        "volume_change_unit": "%",
        "volumetric_energy_density": 863.0,
        "volumetric_energy_density_unit": "Wh/L",
        "volumetric_specific_capacity": 288.2,
        "volumetric_specific_capacity_unit": "mAh/cm^3"
    }


Detailed inputs, outputs, and outline
+++++++++++++++++++++++++++++++++++++
.. aiida-workchain:: VaspCatMatWorkChain
    :module: aiida_catmat.workchains

