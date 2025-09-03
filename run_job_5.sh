#!/bin/sh
#BSUB -q gpua100

### experiment_name
#BSUB -J 62_bsub

### number of core
#BSUB -n 4

### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"

### specify the memory needed
#BSUB -R "rusage[mem=10GB]"

### set walltime limit: hh:mm --
#BSUB -W 23:59

### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."
module load cuda/11.8
module load python3/3.10.13
source /zhome/b5/8/132309/GATMA/venv/bin/activate

### Authenticate wandb
export WANDB_API_KEY=a916b1f26e04fa87565fd5a12282d2b28fede3df

# Sanity check for wandb CLI
echo "🔎 Checking wandb CLI..."
which wandb
wandb --version

cwdpath=$(pwd)
#python3 "$cwdpath/gpu_run.py"
python3 /zhome/b5/8/132309/GATMA/train_5.py

# Attempt to sync wandb logs
echo "Syncing Weights & Biases logs..."
if [ -d "wandb/latest-run" ]; then
    wandb sync wandb/latest-run
    if [ $? -eq 0 ]; then
        echo "wandb sync successful!"
    else
        echo "wandb sync failed — try manually: wandb sync wandb/latest-run"
    fi
else
    echo "No wandb/latest-run directory found — nothing to sync."
fi