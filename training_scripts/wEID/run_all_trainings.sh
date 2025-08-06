#!/usr/bin/env bash
#set -euo pipefail

# Loop over each train script matching the GENIE patterns
for script in train_script_GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_*.py; do
  echo "=== Running $script ==="
  python3 "$script"
done

echo "All training scripts completed."
