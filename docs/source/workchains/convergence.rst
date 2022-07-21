=====================================
VaspConvergeWorkChain
=====================================
The ``VaspConvergeWorkChain`` aims to automate the indentification of 
converged energy cutoff and kpoints mesh for a given structure. In short, it makes
our lives much easier. 

How it works
-------------

The :py:func:`~aiida_catmat.workchains.vasp_converge.VaspConvergeWorkChain` allows us to supply a list of 
``ENCUT`` and ``KSPACING``, provide a structure, hit the button, sit back, and get the converged results as::

    {
        'ENCUT': 500, 
        'KSPACING': 0.4
    }

Under the hood, we use ``S_conv`` protocol with ``VaspMultiStageWorkChain``. It first submits all calculations with
different ``ENCUT`` values in parallel and waits for all to finish. It then, selects the converged ``ENCUT`` by 
considering a threshold::

    {
        "converged_encut": 500,
        "converged_encut_conservative": 550,
        "energy_difference": 0.00023466671880001,
        "final_energy": {
            "500": -357.5085037,
            "550": -357.49348503,
            "600": -357.51247015,
            "650": -357.53935157,
            "700": -357.53571681,
            "750": -357.5267332
        },
        "final_energy_per_atom": {
            "500": -5.5860703703125,
            "550": -5.5858357035937,
            "600": -5.5861323460938,
            "650": -5.5865523682813,
            "700": -5.5864955751562,
            "750": -5.58635520625
        }
    }

Then, it uses the selected ``ENCUT`` (ie, ``converged_encut``) and submits (again in parallel) calculations
with varying ``KSPACING`` and reports the energies::

    {
        "converged_kspacing": 0.4,
        "converged_kspacing_conservative": 0.38,
        "energy_difference": 0.0,
        "final_energy": {
            "0.1": -357.5251117,
            "0.12": -357.52511238,
            "0.14": -357.52513287,
            "0.16": -357.5250471,
            "0.18": -357.52480178,
            "0.2": -357.52480178,
            "0.22": -357.52479079,
            "0.24": -357.52304064,
            "0.26": -357.52297048,
            "0.28": -357.52297048,
            "0.3": -357.5085037,
            "0.32": -357.5085037,
            "0.34": -357.5085037,
            "0.36": -357.5085037,
            "0.38": -357.5085037,
            "0.4": -357.5085037
        },
        "final_energy_per_atom": {
            "0.1": -5.5863298703125,
            "0.12": -5.5863298809375,
            "0.14": -5.5863302010937,
            "0.16": -5.5863288609375,
            "0.18": -5.5863250278125,
            "0.2": -5.5863250278125,
            "0.22": -5.5863248560937,
            "0.24": -5.58629751,
            "0.26": -5.58629641375,
            "0.28": -5.58629641375,
            "0.3": -5.5860703703125,
            "0.32": -5.5860703703125,
            "0.34": -5.5860703703125,
            "0.36": -5.5860703703125,
            "0.38": -5.5860703703125,
            "0.4": -5.5860703703125
        }
    }

and finally reports the final ``Dict`` which we saw first. This can be loaded/linked in any other subsequent
calculations to keep the provenance. 
        

Deailed inputs, outputs, and outline
++++++++++++++++++++++++++++++++++++
.. aiida-workchain:: VaspConvergeWorkChain
    :module: aiida_catmat.workchains

