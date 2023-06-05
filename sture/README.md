

生成并裁剪pdf:

viso2pdf.exe mot\sture\latex\cviu C:\texlive\2016\bin\win32\pdfcrop.exe

由pdf生成eps文件：

C:\Progra~1\Anaconda3\python D:\workspace\dong\ai\tools\latex\pdf2eps.py D:\workspace\dong\mot\sture\latex\cviu



原始可运行的工程路径为：/data2/whd/workspace/MOT/TKP

change dataset

Python 
dataset = 'test' # 'train' or 'test'
detector = 'SDP' # DPM, FRCNN, SDP
frame_path = 'data/MOT17/' + dataset + '/' + seq_name + '-' + detector + '/img1/' + '{:06d}.jpg'.format(frame_id)
frame_path = 'data/MOT16/' + dataset + '/' + seq_name + '/img1/' + '{:06d}.jpg'.format(frame_id)


matlab
globals.m

opt.mot2d = 'MOT16';
opt.mot2d = 'MOT17_SDP';

opt.mot2d = 'MOT20';
opt.results_dir = 'results/';
opt.mot2d_train_seqs = {'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'};
opt.mot2d_train_nums = [429, 2782, 2405, 3315];
opt.mot2d_test_seqs = {'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'};
opt.mot2d_test_nums = [2080, 1008, 585, 806];




cd /data2/whd/workspace/MOT/TKP
nohup /home/d/anaconda2/envs/deepMOT/bin/python /data2/whd/workspace/MOT/TKP/calculateSimilarityTKP.py >output 2>&1 &


NL/net/SimilarityModel.py

pip install tensorflow-gpu==1.9.0 (compatible with CUDA10.1 and cudnn 7.5)
pip install h5py
easy_install pillow


Install:
https://blog.csdn.net/sinat_33486980/article/details/95078922

CUDNN problem:
https://blog.csdn.net/wukai0909/article/details/97489794



{Error using instrument
The instrument object requires JAVA support.

Error in icinterface (line 27)
            obj = obj@instrument(validname);

Error in tcpip (line 71)
            obj = obj@icinterface('tcpip'); %#ok<PROP>
}
{Undefined function or variable 'client_tcp'.
}
{Error using fwrite
Invalid file identifier. Use fopen to generate a valid file identifier.

Error in track_frame (line 109)
            fwrite(client_tcp, 'client ok');

Error in MOT_associate (line 9)
    tracker = track_frame(tracker, fr, frame_image, bboxes_associate,
    index_det, seq_name, opt);

Error in track_seq (line 150)
                trackers{ind} = MOT_associate(fr, frame_image, frame_size,
                bboxes_associate, trackers{ind}, opt, seq_name);
}
{Undefined function or variable 'test_time'.
}
{^HError using fclose
Invalid file identifier. Use fopen to generate a valid file identifier.
}^



# Online Multi-Object Tracking with DMANs

This is the implementation of our ECCV 2018 paper [Online Multi-Object Tracking with Dual Matching Attention Networks](https://arxiv.org/abs/1902.00749). We integrate the ECO [1] for single object tracking. The code framework for MOT benefits from the MDP [2].

<p align="center">
  <img width="800" src="DMAN.png">
</p>
<p align="justify">

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
<pre><code>python calculate_similarity.py
</code></pre>
5. Run the socket client script DMAN_demo.m in Matlab.
# Citation

If you use this code, please consider citing:

<pre><code>@inproceedings{zhu-eccv18-DMAN,
    author    = {Zhu, Ji and Yang, Hua and Liu, Nian and Kim, Minyoung and Zhang, Wenjun and Yang, Ming-Hsuan},
    title     = {Online Multi-Object Tracking with Dual Matching Attention Networks},
    booktitle = {European Computer Vision Conference},
    year      = {2018},
}
</code></pre>

# References
[1] Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: ECO: Efficient convolution operators for tracking. In: CVPR (2017)

[2] Xiang, Y., Alahi, A., Savarese, S.: Learning to track: Online multi-object tracking by decision making. In: ICCV (2015)
