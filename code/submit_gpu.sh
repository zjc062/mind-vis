ts=$(date +%Y-%m-%d_%H-%M-%S)
PROJECT_NAME=mae
BASE_DIR="/home/zijiao/Desktop/Zijiao/HPCresult/${PROJECT_NAME}_${ts}"
CODE_DIR="/home/zijiao/Desktop/Zijiao/side_project/fmri-image-reconstruction"
NUM_GPUS=3
mkdir -p $BASE_DIR
JOB=`/opt/pbs/bin/qsub -V -q gpuQ<<EOJ
#!/bin/bash
#PBS -N ${PROJECT_NAME}
#PBS -l walltime=100:00:00
#PBS -l select=1:ncpus=1:ngpus=${NUM_GPUS}:mem=300gb
#PBS -e "${BASE_DIR}/stderr.txt"
#PBS -o "${BASE_DIR}/stdout.txt"
    export PATH=/mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/Zijiao/anaconda3/bin:$PATH
    source activate /mnt/isilon/CSC4/HelenZhouLab/HZLHD1/Data4/Members/Zijiao/anaconda3/envs/base_zj
    cd ${CODE_DIR}
    # python fmri_pretrain.py
    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 29509 fmri_pretrain.py
EOJ`

echo "${PROJECT_NAME}"
echo "JobID = ${JOB} submitted"
