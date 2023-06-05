solve: AttributeError:module 'cv2.cv2' has no attribute 'DualTVL1OpticalFlow_create' 
pip install opencv_python==3.4.2.17
(NO) pip install opencv-contrib-python==3.4.2.17

# person-reid-lib

The pytorch-based lightweight library of person re-identification.

#### Config

---

Version
```
python 3.6 or 3.7
pytorch >= 0.4
```


Install

```
pip install numpy h5py lmdb
pip install visdom  # Optional. If you don't need a web page visualization, don't install it.
```
Install [pytorch and torchvision](https://pytorch.org/)


Indicates the folder of the original files and where the unzipped file is placed.
```
# person-reid-lib/lib/utils/manager.py
self._device_dict = xxxx
```


#### Optical Flow

---

Install opencv
```
pip install opencv-contrib-python    # version 3.4.2.17
```

Config
```
# person-reid-lib/lib/dataset/utils.py
DataStoreManager.store_optical_flow = True  # if you want to use optical flow, enable it.

# person-reid-lib/tasks/taskname/solver.py
Solver.use_flow = True
```

#### How to run:

---

```
# image-dataset
cd person-reid-lib_folder
sh script/server_0.sh

# video-dataset

cd person-reid-lib_folder
sh script/task_video.sh
```

#### Dataset

---

Image: VIPeR, Market1501, CUHK03, CUHK01, DukeMTMCreID, GRID,

Video : iLIDS-VID, PRID-2011, LPW, MARS, DukeMTMC-VideoReID

#### Updates

---
**2018.12.29**  The code of [Spatial and Temporal Mutual Promotion for Video-based Person Re-identification](https://arxiv.org/abs/1812.10305) is available.

**2018.12.26**  The initial version is available.

**2018.11.19**  The code for *lib* has been released.


#### Related person ReID projects:

---

[deep person reid](https://github.com/KaiyangZhou/deep-person-reid)

[MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation)
