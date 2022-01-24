
README is similar to MMdetection3D github (https://github.com/open-mmlab/mmdetection3d)

## Prerequisites
- CUDA : 10.2
- Cudnn : 7
- pytorch : 1.9.0
- mmcv : 1.4.3
- mmcv-full : 1.4.3
- mmdet : 2.19.0
- mmsegmentation : 0.20.0

I didn't test other version of above packages.

## Install (You are able to pull docker image.)
``` python
sudo docker pull tawn0414/spa_radar_mmdet3d:latest
```
``` python
# Docker run
sudo docker run --gpus all -it -v /home:/home -v /mnt:/mnt --shm-size=512g -p {port}:{port} --name {Container name} docker pull tawn0414/spa_radar_mmdet3d:cuda10.2-torch1.9.0-mmcv1.4.3 /bin/bash

# install SPA_Radar_mmdet3d
git clone https://github.com/mjseong0414/SPA_Radar_mmdet3d.git
cd SPA_Radar_mmdet3d/
pip install -r requirements/build.txt
pip install --no-cache-dir -e .
```


## Install (start from scratch with docker)
1. Docker command
``` python
sudo docker run --gpus all -it -v /home:/home -v /mnt:/mnt --shm-size=512g -p {port}:{port} --name {Container name} pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel /bin/bash
```

2. Package Install
``` python
apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 tmux vim wget && apt-get clean && rm -rf /var/lib/apt/lists/* 
```

3. Install mmcv, mmcv-full, mmdet, mmsegmentation and mmdet3d
``` python
# mmcv
pip install mmcv==v1.4.3

# mmcv-full
pip install mmcv-full==v1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

# mmdet
pip install mmdet==2.19.0

# mmsegmentation
pip install mmsegmentation==0.20.0

# mmdetection3d
git clone https://github.com/mjseong0414/SPA_Radar_mmdet3d.git
cd SPA_Radar_mmdet3d/
pip install -r requirements/build.txt
pip install --no-cache-dir -e .
```