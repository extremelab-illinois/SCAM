# SPDX-License-Identifier: MIT
"""Named gas-composition presets for SCAM chemistry backends."""

PRESETS: dict[str, dict[str, str]] = {
    "air": {
        "edge_x": "O2:0.21,N2:0.79",
        "description": "Standard air edge gas with no pyrolysis stream.",
    },

    "tacot": {
        "edge_x": "O2:0.21,N2:0.79",
        # CH4:0.5551,CO:0.2418,H2O:0.2031 molar → C:0.206 H:0.679 O:0.115
        # (matches PATO tacot26.xml elemental fracs exactly)
        "pyro_y": "CH4:0.5551,CO:0.2418,H2O:0.2031",
        "description": (
            "Air edge gas with TACOT pyrolysis gas "
            "(tacot26.xml: C:0.206 H:0.679 O:0.115 elemental mass fracs). "
            "Reproduces the TACOT reference B' table setup."
        ),
    },

    "co2-n2": {
        "edge_x": "CO2:0.96,N2:0.04",
        "description": "Mars-atmosphere proxy: 96% CO2, 4% N2 molar, no pyrolysis.",
    },

    "silica-sio": {
        "edge_x": "O2:1.0",
        "target_element": "Si",
        "gas_mechanism": "mechanisms/sio_silica.yaml",
        "gas_phase_name": "sio_gas",
        "carbon_phase": "mechanisms/sio_silica.yaml",
        "condensed_phase_name": "silica_condensed",
        "description": "Minimal Si/O silica ablation preset.",
    },

    "silica-sionc": {
        "edge_x": "O2:0.21,N2:0.79",
        "pyro_y": "CO:0.5,CO2:0.5",
        "target_element": "Si",
        "gas_mechanism": "mechanisms/sionc_silica.yaml",
        "gas_phase_name": "sionc_gas",
        "carbon_phase": "mechanisms/sionc_silica.yaml",
        "condensed_phase_name": "silica_condensed",
        "description": "Minimal Si/O/N/C silica ablation preset.",
    },
}
