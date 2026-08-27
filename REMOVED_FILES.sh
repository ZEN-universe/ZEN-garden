#!/bin/bash
# ZEN-garden docs restructure: files that must be DELETED.
# Claude could not delete files on your machine (no delete permission on the
# connected folder), so this is the one manual step. Run from the repo root.
set -e

# --- tutorials that were renamed or whose content moved elsewhere ---
git rm docs/files/tutorial/00_tutorial_overview.rst
git rm docs/files/tutorial/01_analyze_outputs.rst
git rm docs/files/tutorial/03_add_technologies_carrier.rst
git rm docs/files/tutorial/04_scenario_analysis.rst        # -> zen_garden_in_detail/scenario_tool.rst
git rm docs/files/tutorial/05_time_series_aggregation.rst  # -> split, see below
git rm docs/files/tutorial/07_scaling.rst                  # -> zen_garden_in_detail/scaling.rst
git rm docs/files/tutorial/08_operation_only.rst
git rm docs/files/tutorial/09_handle_infeasibilities.rst
git rm docs/files/tutorial/10_troubleshooting.rst          # -> support/troubleshooting.rst
git rm docs/files/tutorial/tables/benchmarking_model.csv   # -> zen_garden_in_detail/tables/

# --- PWA removal (feature no longer in the model) ---
git rm -r docs/dataset_examples/4_PWA_nonlinear_capex
git rm docs/files/figures/zen_garden_in_detail/PWA.png

# --- superseded legacy tutorial (not in any toctree, never rendered) ---
# References system.py, the retired RRE-ETH repo, a dead branch URL, and a
# third incompatible dataset numbering. Its myopic-foresight comparison has
# been folded into docs/files/tutorial/12_myopic_foresight.rst.
git rm docs/dataset_examples/dataset_creation_tutorial.md
