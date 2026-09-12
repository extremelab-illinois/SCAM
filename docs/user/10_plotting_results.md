<!-- SPDX-License-Identifier: MIT -->
# Plotting results

`examples/templates/plot_results.py` is a standalone plotter that reads any
SCAM CSV results directory and produces a static 3-panel PNG, plus two
optional animation modes. It needs `MPLBACKEND=Agg` in a headless
environment (no display) to avoid blocking on `plt.show()`.

```bash
# Static PNG only (always produced)
python3 examples/templates/plot_results.py results/tacot_arcjet_template/

# PNG + video (mp4 if ffmpeg is available, otherwise gif)
python3 examples/templates/plot_results.py results/tacot_arcjet_template/ --video

# Control video frame rate and speed
python3 examples/templates/plot_results.py results/ablation1_template/ --video --fps 20 --speed 4

# Slab animation
python3 examples/templates/plot_results.py results/ablation2_template/ --slab --speed 30
```

## Options

| Flag | Default | Notes |
|---|---|---|
| `results_dir` | — | Positional; path to a SCAM CSV output directory. |
| `--video` | off | 1-D temperature-profile animation, sweeping with depth. Red dashed line = receding surface; blue dashed line = char-fraction isoline at `--beta`. |
| `--slab` | off | 2-D slab visualization: the 1-D profiles are extruded to a thin slab and rendered as colored `imshow` panels — temperature, total char fraction β, and β per Arrhenius component. Ablated zone shown in light gray. Requires `comp{c}_beta_profiles.csv`, which the CLI writes automatically alongside `char_fraction_profiles.csv` (CSV output only — see [`09_output_files.md`](09_output_files.md)). |
| `--fps N` | `15` | Frames per second for the video. |
| `--speed N` | auto | Simulation seconds per real second of video; overrides `--fps`. |
| `--beta LEVEL` | `0.01` | Char-fraction isoline drawn on `--video` (default: the leading edge of decomposition). |

```{note}
The script's own module docstring lists only `--video`, `--fps`, and
`--speed` under "Options" — `--slab` and `--beta` exist and work (see the
argparse definitions in the script) but are undocumented there. The table
above is complete; treat it as authoritative over the script's docstring
until that's fixed.
```

Both animation modes use `depth_profiles.csv` (not the depth column baked
into `temperature_profiles.csv`) as the depth axis, which matters under ALE
recession — see [`09_output_files.md`](09_output_files.md).

Output is written as `.mp4` if `ffmpeg` is available on the system,
otherwise as `.gif`.

## Cleaning up generated plots

`examples/clean_outputs.py` removes every generated PNG, `results_*/`
directory, and `__pycache__/` under `examples/`:

```bash
python3 examples/clean_outputs.py
python3 examples/clean_outputs.py --dry-run   # preview without deleting
```
