# SPDX-License-Identifier: MIT
"""Plot digitized BC data from Amar (2006) Figures 8.14 and 8.15.

Run:
    MPLBACKEND=Agg python3 examples/verification/amar_thesis/plot_bc_figs14_15.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

BTU_LBM_TO_J_KG   = 2326.0
LBM_FT2S_TO_KG_M2S = 4.8824
ATM_TO_PA          = 101325.0
R_TO_K             = 5.0 / 9.0


def load(name):
    p = HERE / name
    d = np.loadtxt(str(p), delimiter=",")
    order = np.argsort(d[:, 0])
    return d[order, 0], d[order, 1]


t_p,   p_atm   = load("figure8.14_pressure.csv")
t_hr,  hr_btu  = load("figure8.14_recovery_enthalpy.csv")
t_htc, htc_lbm = load("figure8.15_heat_transfer_coefficient.csv")  # HTC [lbm/ft²·s]
t_ev,  ev_ftps = load("figure8.15_edge_velocity.csv")               # edge velocity [ft/s]

# SI conversions
p_Pa   = np.maximum(p_atm,  0.0) * ATM_TO_PA
hr_Jkg = np.maximum(hr_btu, 0.0) * BTU_LBM_TO_J_KG
htc_SI = np.maximum(htc_lbm, 0.0) * LBM_FT2S_TO_KG_M2S

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# --- Fig 8.14: Pressure ---
ax = axes[0, 0]
ax.plot(t_p, p_atm, 'b-', lw=1.5)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Pressure [atm]")
ax.set_title("Fig 8.14 — Edge pressure")
ax.grid(True, alpha=0.3)
peak_idx = np.argmax(p_atm)
ax.annotate(f"peak {p_atm[peak_idx]:.2f} atm\nt={t_p[peak_idx]:.1f} s",
            xy=(t_p[peak_idx], p_atm[peak_idx]),
            xytext=(t_p[peak_idx]+3, p_atm[peak_idx]*0.8),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')

# --- Fig 8.14: Recovery enthalpy ---
ax = axes[0, 1]
ax.plot(t_hr, hr_btu, 'r-', lw=1.5, label="[Btu/lbm]")
ax2 = ax.twinx()
ax2.plot(t_hr, hr_Jkg / 1e6, 'r--', lw=1.0, alpha=0.5, label="[MJ/kg]")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Recovery enthalpy [Btu/lbm]")
ax2.set_ylabel("Recovery enthalpy [MJ/kg]", color='r')
ax.set_title("Fig 8.14 — Recovery enthalpy h_r")
ax.grid(True, alpha=0.3)
ax.annotate(f"t=23.5 s\nlaminar→turbulent",
            xy=(23.5, float(np.interp(23.5, t_hr, hr_btu))),
            xytext=(30, 6000),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray')

# --- Fig 8.14: Convective power = HTC × h_r ---
ax = axes[0, 2]
htc_at_thr = np.interp(t_hr, t_htc, htc_SI)
q_conv = htc_at_thr * hr_Jkg / 1e6   # [MW/m²]
ax.plot(t_hr, q_conv, 'm-', lw=1.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("q_conv = HTC × h_r  [MW/m²]")
ax.set_title("Derived: convective heat flux (unblown)")
ax.grid(True, alpha=0.3)
peak_idx = np.argmax(q_conv)
ax.annotate(f"peak {q_conv[peak_idx]:.1f} MW/m²\nt={t_hr[peak_idx]:.1f} s",
            xy=(t_hr[peak_idx], q_conv[peak_idx]),
            xytext=(t_hr[peak_idx]+3, q_conv[peak_idx]*0.7),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')

# --- Fig 8.15: HTC ---
ax = axes[1, 0]
ax.plot(t_htc, htc_lbm, 'b-', lw=1.5, label="[lbm/ft²·s]")
ax2 = ax.twinx()
ax2.plot(t_htc, htc_SI, 'b--', lw=1.0, alpha=0.5, label="[kg/m²·s]")
ax.set_xlabel("Time [s]")
ax.set_ylabel("HTC ρ_e u_e C_H  [lbm/ft²·s]")
ax2.set_ylabel("HTC  [kg/m²·s]", color='b')
ax.set_title("Fig 8.15 — Heat transfer coefficient [lbm/ft²·s]")
ax.grid(True, alpha=0.3)
peak_idx = np.argmax(htc_lbm)
ax.annotate(f"peak {htc_lbm[peak_idx]:.3f} lbm/ft²·s\nt={t_htc[peak_idx]:.1f} s",
            xy=(t_htc[peak_idx], htc_lbm[peak_idx]),
            xytext=(t_htc[peak_idx]+3, htc_lbm[peak_idx]*0.7),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')

# --- Fig 8.15: Edge velocity ---
ax = axes[1, 1]
ax.plot(t_ev, ev_ftps / 1000, 'g-', lw=1.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Edge velocity [×10³ ft/s]")
ax.set_ylim(bottom=0)
ax.set_title("Fig 8.15 — Edge velocity [ft/s]")
ax.grid(True, alpha=0.3)
ax.annotate(f"start {ev_ftps[0]/1000:.1f}×10³ ft/s",
            xy=(t_ev[0], ev_ftps[0]/1000),
            xytext=(8, ev_ftps[0]/1000*0.75),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray')

# --- Laminar/turbulent indicator ---
ax = axes[1, 2]
ax.plot(t_htc, htc_lbm, 'b-', lw=2, label="HTC [lbm/ft²·s]")
ax.plot(t_p,   p_atm,   'k-', lw=1.5, label="Pressure [atm]")
ax.axvline(23.5, color='orange', ls='--', lw=1.5, label="Transition t≈23.5 s")
ax.set_xlabel("Time [s]")
ax.set_ylabel("HTC [lbm/ft²·s]  /  Pressure [atm]")
ax.set_title("BC overview: HTC + pressure")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Amar (2006) §8.7/§8.8 — Digitized boundary conditions\n"
    "Figs 8.14 (pressure, h_r) and 8.15 (HTC, edge velocity)",
    fontsize=12,
)
fig.tight_layout()

out = HERE / "plot_bc_figs14_15.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
