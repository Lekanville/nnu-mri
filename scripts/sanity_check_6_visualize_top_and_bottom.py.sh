#!/bin/bash

#SBATCH --job-name=visualize_top_and_bottom_nnu_predictions
#SBATCH --output=visualize_top_and_bottom.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                # 16GB is plenty for loading 3D volumes
#SBATCH --time=00:30:00          # This will only take about 5-10 minutes

source ~/miniforge3/bin/activate nnu_env

# Run the python script (save your python code as visualize_top_and_bottom.py)
python sanity_check_5_visualize_top_and_bottom.py