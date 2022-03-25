[![arXiv](https://img.shields.io/badge/arXiv-2103.00738-b31b1b.svg)](https://arxiv.org/abs/2103.00738)
# FPS-Net
Code for "FPS-Net: A convolutional fusion network for large-scale LiDAR point cloud segmentation", accepted by [ISPRS journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/abs/pii/S092427162100112X)  
By [Aoran Xiao](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=zh-EN), Xiaofei Yang, Shijian Lu, Dayan Guan, Jiaxing Huang  

[Full Paper](https://arxiv.org/pdf/2103.00738.pdf)

## Install
```
conda create -n FPSNet python=3.7
source activate FPSNet
cd /ROOT/
pip install -r requirements.txt
```

## Train
```
cd /train/tasks/semantic
sh train.sh
```

## Inference and Test
```
cd /train/tasks/semantic
sh test.sh
```

## Citation
If you use this code, please cite:
```
@article{xiao2021fps,
  title={FPS-Net: A convolutional fusion network for large-scale LiDAR point cloud segmentation},
  author={Xiao, Aoran and Yang, Xiaofei and Lu, Shijian and Guan, Dayan and Huang, Jiaxing},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={176},
  pages={237--249},
  year={2021},
  publisher={Elsevier}
}
```
## Acknowledgement
Part of code is borrowed from [lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal), thanks for their sharing!
## Related Repos
- [SynLiDAR: Learning From Synthetic LiDAR Sequential Point Cloud for Semantic Segmentation](https://github.com/xiaoaoran/SynLiDAR)
- [Unsupervised Representation Learning for Point Clouds: A Survey](https://github.com/xiaoaoran/3d_url_survey)
