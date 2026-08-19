# MRI Image Segmentation using nnU-Net
A reproducible pipeline for 3D medical image segmentation using the state-of-the-art **nnU-Net** framework. This project serves as a technical validation baseline for downstream whole-body adipose tissue segmentation and radiomic feature analysis.

---

## Prerequisites

* **Conda / Mamba** (for environment management)
* **CUDA-compatible GPU** (recommended for running training and inference)

---

## Setup & Installation

Clone the repository and set up the environment using the provided configuration file:

```bash
# Clone the repository
git clone [https://github.com/Lekanville/nnu-mri.git](https://github.com/Lekanville/nnu-mri.git)
cd nnu-mri

# Create and activate the Conda environment
conda env create -f environment.yml
conda conda activate adipose_seg  # (or your chosen environment name)
```

## Configuration
nnU-Net requires specific environment paths to locate raw, preprocessed, and trained model directories. Add these to your shell profile (e.g. slurm, ~/.bashrc or ~/.zshrc):

export nnUNet_raw="/path/to/your/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/nnUNet_trained_models"

## Folder Structure
The repository is structured to separate code and execution scripts from heavy data and trained model weights (which are ignored via .gitignore):
```text
mri_nnu/
│
├── scripts/                      # Core execution and utility scripts
│   ├── 3d_fullres/               # Configuration files for full-resolution 3D models
│   │   ├── 2_nnu_3d_train_fold_0.sh
│   │   ├── 2_nnu_3d_train_fold_1.sh
│   │   ├── 2_nnu_3d_train_fold_2.sh
│   │   ├── 2_nnu_3d_train_fold_3.sh
│   │   └── 2_nnu_3d_train_fold_4.sh
│   ├── 1_nnu_plan_and_preprocess.sh
│   ├── 3_nnv_3d_best_conf.sh
│   ├── 4_nnu_3d_predict.sh
│   ├── 5_nnu_postprocessing.sh
│   └── sanity_check_*.py         # Diagnostic and validation scripts
│
├── environment.yml               # Conda environment specifications
├── .gitignore                    # Ignores large datasets and model weights
└── README.md
```

## Pipeline Execution
* Plan and Preprocess:
```bash
sbatch scripts/1_nnu_plan_and_preprocess.sh
```

* Train the Model:
```bash
sbatch scripts/3d_fullres/2_nnu_3d_train_fold_0.sh
sbatch scripts/3d_fullres/2_nnu_3d_train_fold_1.sh
sbatch scripts/3d_fullres/2_nnu_3d_train_fold_2.sh
sbatch scripts/3d_fullres/2_nnu_3d_train_fold_3.sh
sbatch scripts/3d_fullres/2_nnu_3d_train_fold_4.sh
```

* Run Best Configuration:
```bash
sbatch scripts/3_nnv_3d_best_conf.sh
```

* Run Inference:
```bash
sbatch scripts/4_nnu_3d_predict.sh
```

* Evaluation:
```bash
sbatch scripts/scripts/sanity_check_***.py
```

## License & Acknowledgements
Built on top of the nnU-Net framework created by the Applied Computer Vision Lab (ACVL) and the Division of Medical Image Computing at the German Cancer Research Center (DKFZ) in Heidelberg. Developed as part of a pilot study for MRI adipose segmentation.
