# mmsegmentation-ros

This is a ROS package for segmentation, which utilizes the toolbox [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) of [OpenMMLab](https://openmmlab.com/).

## Requirements

- ROS Melodic
- Python 3.6+, PyTorch 1.3+, CUDA 9.2+ and [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation

1. Clone packages

   ```bash
   git clone https://github.com/jianhengLiu/mmsegmentation_ros.git 
   cd mmdetection_ros
   git clone https://github.com/open-mmlab/mmsegmentation.git
   ```

2. `MMSegmentation` Requirements, please refer to <https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation>

   every time you change the path of `mmsegmentation` must not forget to run

   ```bash
   cd mmsegmentation
   pip install -e .
   ```

3. Install `rospkg`.

   ```bash
   pip install rospkg
   ```

