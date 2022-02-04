#!/bin/bash
set -euxo pipefail

# https://github.com/nat-chan/bizarre-pose-estimator#download
gdrive download --recursive 11bw47Vy-RPKjgd6yF0RzcXALvp7zB_wt
unzip wacv2022_bizarre_pose_estimator_release/bizarre_pose_models.zip
rsync -av ./bizarre_pose_models/ ./
rm -rf bizarre_pose_models
unzip wacv2022_bizarre_pose_estimator_release/bizarre_pose_dataset.zip
mv bizarre_pose_dataset _data
unzip wacv2022_bizarre_pose_estimator_release/character_bg_seg_data.zip
mv character_bg_seg _data
rm -rf wacv2022_bizarre_pose_estimator_release

# https://github.com/nat-chan/bizarre-pose-estimator#setup
cp ./_env/machine_config.bashrc.template ./_env/machine_config.bashrc
sed -i "s;PROJECT_DN=;PROJECT_DN=$(readlink -f .);" _env/machine_config.bashrc

chmod +x make/*
./make/docker_pull
./make/shell_docker