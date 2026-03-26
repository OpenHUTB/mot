# Non-local Attention Association Scheme for Online Multi-Object Tracking

This is the implementation of the paper [Non-local attention association scheme for online multi-object tracking](https://doi.org/10.1016/j.imavis.2020.103983). We integrate non-local attention [1] for multi-object tracking. The code framework for MOT benefits from the MDP [2].

# Prerequisites
- Cuda 8.0
- Cudnn 5.1
- Python 2.7
- Keras 2.0.5
- Tensorflow 1.1.0

For example:
<pre><code>conda create -n mot anaconda python=2.7
conda activate mot
conda install -c menpo opencv
pip install tensorflow-gpu==1.1.0
pip install keras==2.0.5
</code></pre>

# Usage
1. Download the [DMAN model](https://zhiyanapp-build-release.oss-cn-shanghai.aliyuncs.com/zhuji_file/spatial_temporal_attention_model.h5) and put it into the "model/" folder.
2. Download the [MOT16 dataset](https://motchallenge.net/data/MOT16/), unzip it to the "data/" folder.
3. Cd to the "ECO/" folder, run the script install.m to compile libs for the ECO tracker
4. Run the socket server script:
<pre><code>cd nlaa
python calculate_similarity.py
</code></pre>
5. Run the socket client script DMAN_demo.m in Matlab.
# Citation

If you use this code, please consider citing:

<pre><code>@article{nlaa,
	author={Haidong Wang and Saizhou Wang and Jingyi Lv and Chenming Hu and Zhiyong Li},
	title={Non-local Attention Association Scheme for Online Multi-Object Tracking.},
    journal={Image and Vision Computing},
	volume=102,
	year=2020,
}
</code></pre>

# References
[1] Zhang, Yulun et al. "Residual Non-local Attention Networks for Image Restoration.",ICLR 2019 (2019)

[2] Xiang, Y., Alahi, A., Savarese, S.: Learning to track: Online multi-object tracking by decision making. In: ICCV (2015)
