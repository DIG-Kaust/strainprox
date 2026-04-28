#!/bin/bash -l
#SBATCH --job-name=strain_grid
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-299


conda activate strainprox2

# ---- Grid definition ----
MU_VALUES=(0.1 0.3 0.6 1.0 10)
TAU_SEG_VALUES=(0.5 1.0 2.0)
ALPHA_VALUES=(1.0)
BETA_VALUES=(1.0)
DELTA_VALUES=(0.1 1.0 2.0 5.0 10.0)
NITER_VALUES=(2)
PDNITER_VALUES=(10)
SEGNITER_VALUES=(10 20)
BISNITER_VALUES=(5 10)
CL_VALUES=(0.25)


N_MU=${#MU_VALUES[@]}
N_TAU_SEG=${#TAU_SEG_VALUES[@]}
N_ALPHA=${#ALPHA_VALUES[@]}
N_BETA=${#BETA_VALUES[@]}
N_DELTA=${#DELTA_VALUES[@]}
N_NITER=${#NITER_VALUES[@]}
N_PD=${#PDNITER_VALUES[@]}
N_SEG=${#SEGNITER_VALUES[@]}
N_BIS=${#BISNITER_VALUES[@]}
N_CL=${#CL_VALUES[@]}

TOTAL=$((N_MU * N_TAU_SEG * N_ALPHA * N_BETA * N_DELTA * N_NITER * N_PD * N_SEG * N_BIS * N_CL))
echo "Total combinations: ${TOTAL}  (set --array=0-$((TOTAL-1)))"

# ---- Decode flat index into per-parameter indices ----
IDX=${SLURM_ARRAY_TASK_ID}
i_mu=$((IDX % N_MU));      IDX=$((IDX / N_MU))
i_tau_seg=$((IDX % N_TAU_SEG)); IDX=$((IDX / N_TAU_SEG))
i_alpha=$((IDX % N_ALPHA)); IDX=$((IDX / N_ALPHA))
i_beta=$((IDX % N_BETA));  IDX=$((IDX / N_BETA))
i_delta=$((IDX % N_DELTA)); IDX=$((IDX / N_DELTA))
i_niter=$((IDX % N_NITER)); IDX=$((IDX / N_NITER))
i_pd=$((IDX % N_PD));      IDX=$((IDX / N_PD))
i_seg=$((IDX % N_SEG));    IDX=$((IDX / N_SEG))
i_bis=$((IDX % N_BIS));    IDX=$((IDX / N_BIS))
i_cl=$((IDX % N_CL));      IDX=$((IDX / N_CL))

MU=${MU_VALUES[$i_mu]}
TAU_SEG=${TAU_SEG_VALUES[$i_tau_seg]}
ALPHA=${ALPHA_VALUES[$i_alpha]}
BETA=${BETA_VALUES[$i_beta]}
DELTA=${DELTA_VALUES[$i_delta]}
NITER=${NITER_VALUES[$i_niter]}
PDNITER=${PDNITER_VALUES[$i_pd]}
SEGNITER=${SEGNITER_VALUES[$i_seg]}
BISNITER=${BISNITER_VALUES[$i_bis]}
CL=${CL_VALUES[$i_cl]}

echo "Task ${SLURM_ARRAY_TASK_ID}: mu=${MU} tau_seg=${TAU_SEG} alpha=${ALPHA} beta=${BETA} delta=${DELTA} cl=${CL} niter=${NITER} pdniter=${PDNITER} segmentniter=${SEGNITER} bisectniter=${BISNITER}"

# ---- Setup ----
cd ${SLURM_SUBMIT_DIR}
mkdir -p logs
mkdir -p ../results/hess/grid_search4

# ---- Run ----
python loop.py \
    --mu ${MU} \
    --tau_seg ${TAU_SEG} \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --delta ${DELTA} \
    --cl ${CL} \
    --niter ${NITER} \
    --pdniter ${PDNITER} \
    --segmentniter ${SEGNITER} \
    --bisectniter ${BISNITER} \
    --outdir ../results/hess/grid_search4
