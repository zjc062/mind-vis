cd ./mind-vis

# download data and model checkpoints
bash ./scripts/download_data.sh

python ./code/stageA1_mbm_pretrain.py
