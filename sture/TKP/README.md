## [Temporal Knowledge Propagation for Image-to-Video Person Re-identification](https://arxiv.org/abs/1908.03885)

#### Requirements: Python=3.6 and Pytorch=1.0.0

  ```Shell
  torch 1.3
  torchvision 0.4
  --root /data2/whd/workspace/MOT/TKP/TKP/data -d mars --save_dir data/log/mars/ --gpu_devices 0,1 --train_batch 8
  ```


### Training and test

  ```Shell
  # For MARS
  python train.py --root /data2/whd/workspace/MOT/TKP/TKP/data -d mars --save_dir /data3/dong/data/tkp/log/mars/ --train_batch 8 --gpu_devices 0

  python test.py --root /data2/whd/workspace/MOT/TKP/TKP/data -d mars --resume /data3/dong/data/tkp/log/mars/best_model.pth.tar --save_dir /data3/dong/data/tkp/log/mars/ --gpu_devices 1
  
  # 198利用MARS数据集训练
  nohup python train.py --root /data2/whd/workspace/mot/mot/sture/TKP/data -d mars --save_dir /data2/whd/workspace/mot/mot/sture/TKP/data/log/mars/ --train_batch 32 --gpu_devices 0 &

  # 198上利用DukeMTMC-VideoReID数据集训练
  nohup python train.py --root /data2/whd/workspace/mot/mot/sture/TKP/data -d dukevid --save_dir /data2/whd/workspace/mot/mot/sture/TKP/data/log/duke --gpu_devices 0 &

  python train.py --root /data/datasets/ -d dukevid --save_dir log-duke
  python test.py --root /data/datasets/ -d dukevid --resume log-duke/best_model.pth.tar --save_dir log-duke
  
  # For iLIDS-VID (If you use the pretrained model on Duke, you will get a much higher results than that reported in our paper.)
  python main_ilids.py --root /data/datasets/ --save_dir log-ilids
  ```

# 画图
1.分别指定test.py的resume参数为训练第一部和训练最好的模型；
```buildoutcfg
--resume /data3/dong/data/tkp/log/mars/best_model.pth.tar
```

2.调试运行test.py，
```buildoutcfg
for batch_idx, (vids, pids, camids) in enumerate(queryloader):
    if (batch_idx+1)%1000==0 or (batch_idx+1)%len(queryloader)==0:
        print("{}/{}".format(batch_idx+1, len(queryloader)))

    vid_qf.append(extract_vid_feature(vid_model, vids, use_gpu).squeeze())
    vid_q_pids.extend(pids)
    vid_q_camids.extend(camids)
```
改循环运行至少100遍，然后运行一下代码保存数据给Matlab调用：
```buildoutcfg
scipy.io.savemat('vid_qf.mat', {'vid_qf': torch.stack(vid_qf).numpy()})
scipy.io.savemat('ids.mat',
                 {'vid_q_pids': torch.stack(vid_q_pids).numpy(),
                  'vid_q_camids': torch.stack(vid_q_camids).numpy()})
```
3.在Matlab中执行tsnet_plot.m用于压缩每个video的特征到2维并绘制图像，保存显示结果。 


### Citation

If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @inproceedings{gu2019TKP,
      title={Temporal Knowledge Propagation for Image-to-Video Person Re-identification},
      author={Gu, Xinqian and Ma, Bingpeng and Chang, Hong and Shan, Shiguang and Chen, Xilin},
      booktitle={ICCV},
      year={2019},
    }