cd ./mind-vis

# download data and model checkpoints
bash ./scripts/download_data.sh

python ./code/stageA1_mbm_pretrain.py
python ./code/stageA2_mbm_train.py
python ./code/stageB1_mbm_pretrain.py