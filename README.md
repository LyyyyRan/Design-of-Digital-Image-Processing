# Design-of-Digital-Image-Processing

# 简介
    An project based on YOLOv5-WiderFace && MobileNetV3 for course design.

# 环境依赖
    1) Request python3
 
# 目录结构描述
    ├── main.py             // Main Work written in python
    
    ├── models              // some {modules}.py && {descript}.yaml

    ├── checkpoint          // some {weights}.pt
    
    │   ├── weights         // MobileNetV3-FaceKeypoints

    │   ├── det_weights     // YOLOv5-S-WiderFace

    ├── utils               // some {utilities}.py

    ├── predict.py          // An Pipeline between 2 Main Models

    └── ReadMe.md           // Introduction

# 使用说明
    1) Run main.py after the needed pkgs are all installed.
    
    2) [Open Camera] 
        1. open && start to read camera: vdieo0;
        2. close window showing image && release camera;

    3) [Exit]: exit() in the python script.
    
    4) [Detect]
        1. start the 2 models to predict;
        2. stop predict;

# 版本内容更新
###### v0:
    1. 实现 YOLOv5 人脸目标检测；
    2. 实现 MobileNet 人脸关键点定位；
###### v1:
    for 大创：
        1. 基于 ROS::CV_Bridge 实现进程间图像共享；
        2. 基于 ROS::Topic 实现进程间数据共享；
###### v2(Now):
    for 数字图像处理课设：
        1. 增加了基于 PyQT 实现界面设计控制程序；
