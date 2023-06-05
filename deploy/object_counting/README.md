<h1 align='center'>
多目标跟踪计数
</h1>

一个 Flask 应用程序，用于通过网络进行多个实时视频流，具有对象检测、跟踪（可选）和计数功能。 
使用带有 Tensorflow 后端的 YOLO v4 作为对象检测模型，并使用在 MARS 数据集上训练的 Deep SORT 进行对象跟踪。 
每个视频流都有一个独立的线程，并使用 ImageZMQ 进行帧的异步发送和处理。



这个应用程序的大部分工作方式是基于 Miguel Grinberg 的 [Flask 视频流](https://github.com/miguelgrinberg/flask-video-streaming)。 
由于这是一个 Flask 应用程序，视频流是通过网络浏览器访问的，并且可能有多个网络客户端。 
如果所有 Web 客户端都与应用程序断开连接，则每个视频流线程将在设定的时间段后由于不活动而自动关闭。 
一旦网络客户端再次连接，视频流将重新启动，但与 Miguel 的应用程序不同，发送帧的摄像头客户端必须重新启动。

[流程解读](https://blog.csdn.net/woshicver/article/details/111878201)

[代码分析](https://blog.csdn.net/weixin_44879270/article/details/106192706)

# TODO
* 命名 
    * 阿耳戈斯  外文名Argus别    名百眼巨人
    * 烛龙
    * 火眼狻猊
* 视频回放

* 使用gpu
* 界面支持中文


# 环境安装
有问题
```shell script
conda env create -f environment.yml  -n mot_web
```

## tensorlfow
必须使用cuda 10.0（默认/usr/local/cuad软链接指向它），否则gpu使用不了。

***
## 摄像头客户端设置
在每个客户端脚本中输入自己的摄像头流地址，并确保每个摄像头客户端都将帧发送到正确的服务器地址和端口。

如果要添加另一个摄像头，需要创建一个带有新端口的 camera_client_2.py 并将此端口添加到 app.py 中的 `port_list` 作为第三个端口。

无论希望显示多少，templates/index.html需要确保  文件包含正确数量的流。

例如，如果要激活第二个摄像头和 YOLO 流，请确保将这些注释取消：

```shell script
<img src="{{ url_for('video_feed', feed_type='camera', device=1) }}"...
```
并且取消注释：
```
<img src="{{ url_for('video_feed', feed_type='yolo', device=1) }}"...
```

可以按照相同的模式使用 device=2 添加更多流，依此类推。

如果想了解有关服务器和客户端如何使用 ZeroMQ 的更多信息，请参阅！[ImageZMQ](https://github.com/jeffbass/imagezmq)。 
当前使用的默认消息传递模式是请求-回复模式。


### 改变相机流的名字
在摄像头客户端文件中编辑“cam_id”，来更改左上角显示的每个摄像头流的名称。


***
## 模型
此应用使用从 Darknet 转换为 Keras 格式的 YOLO v4 权重。 
需要自己训练或转换并将其放在 `model_data` 文件夹中。 
具体操作参阅此 [存储库](https://github.com/Ma-Dan/keras-yolo4)。

可以在 yolo.py 修改 IOU 阈值、锚点、类名等检测参数。


### 检测多个类别
如果要检测多个类别，需要修改 `yolo.py` 中的第 103 行。 
例如，要检测人和汽车，将其更改为：
```
if predicted_class != 'person' and predicted_class != 'car':
    continue
```

***
## 深度排序跟踪算法
可以使用深度排序进行目标跟踪。 
但是，注意跟踪模型仅针对跟踪人进行训练，因此需要自己训练模型以跟踪其他对象。 
正如您在演示图片中看到的那样，它仍然可以用于跟踪其他目标，例如汽车。

请参考 [余弦度量学习](https://github.com/nwojke/cosine_metric_learning) 来训练自己的深度排序跟踪模型。


### 使用深度排序跟踪算法失效
但是，如果不想使用跟踪，可以通过将 camera_yolo.py 的第 28 行从
```
tracking = True
```
改为
```
tracking = False
```

***
# 运行
## 步骤
1.运行 `app.py`、`camera_client_0.py` 启动服务；
或者使用 `video_streamer/video_streamer.py` 和视频文件模拟 IP 相机。

图片转视频的命令：
```-f: 强制输入输出文件的格式（format）；-i：输入（input）文件的地址
ffmpeg -f image2 -i %d.png video.mp4
ffmpeg -f image2 -framerate 20 -i %d.png video.mp4
```

修改同时运行的相机数目，需要修改 `app.py` 中 `video_feed` 中的 `port_list = (5555, 5566)  # 5577`、`templates/index.html` 中的 `<div   </div>`、`video_sreamer/video_streamer.py` 中的 `__name__`。

2.浏览器访问
```shell script
http://115.157.195.140:5000
```

## 调试
`app.py` 和 `camera_client_0.py` 必须同时为调试模式，否则在断点处不会停止。



## 本地运行
首先，在 `camera_client_0.py` 中配置每个摄像头客户端，使其拥有正确的摄像头流地址。 
还要确保使用正确的端口（例如 tcp://localhost:5555）将帧发送到 localhost。 
如前所述，还可以通过将 `cam_id` 更改为更相关的名称来更改相机流的名称。

运行前，检查 templates/index.html 是否配置正确。


先运行 `app.py`，然后开始运行每个相机客户端。 
一切运行后，启动 Web 浏览器并输入 localhost:5000。 
这应该在浏览器中打开 index.html 并且视频线程应该开始运行。 
相机流应该很快加载； 如果没有，请尝试重新启动相机客户端并刷新浏览器。 
YOLO 流最终也会加载，但由于启动 Tensorflow 和加载 YOLO 模型，它们需要更长的时间来加载。


如果 YOLO 线程在加载完成之前就关闭了，则需要在 `base_camera.py` 的第 117 行增加时间限制：
```
if time.time() - BaseCamera.last_access[unique_name] > 60:
```
如果应用程序认为不再有查看 Web 客户端，所有线程将在此时间限制后关闭。 
如果不使用 YOLO 流，则相机流线程关闭的默认时间限制为 5 秒。


***
## 远程运行
该过程类似于在本地运行。 
如果有自己的远程服务器，需要配置每个摄像头客户端，以便它们使用正确的端口而不是 localhost 向该服务器地址发送帧（例如 tcp://server-address-here:5555）。

克隆远程服务器上的存储库并检查它是否转发了正确的端口，以便浏览器和相机客户端可以连接到它。 
运行 `app.py`，然后开始运行相机客户端。 
像之前一样，现在应该能够通过在浏览器中输入端口为 5000 的服务器地址来连接到应用程序（即将 localhost:5000 替换为 server-address-here:5000）。 
同样，如果没有加载，请尝试重新启动相机客户端并刷新浏览器。

***
# 功能
## 计数
对于每个检测到的目标，当前目标的总计数会自动存储在一个文本文件中，每隔一小时设置一次。 
每个新检测到的类还会创建一个新的类计数文件来存储该类的当前计数，并且还将在 YOLO 流中显示为文本。


***
## 性能
使用的硬件：
* Nvidia GTX 1070 GPU
* i7-8700K CPU

将 640x480 分辨率的单个摄像机流在本地以 30FPS 的速度在本地流式传输时，在本地服务器上托管平均可提供约 15FPS。 
关闭跟踪为 16FPS。 
拥有多个流将显着降低 FPS，如演示 gif 所示。

还有很多其他因素会影响性能，例如网络速度和带宽。

降低流的分辨率或质量将提高性能，但也会降低检测精度。
 


# 问题
* 相机图像在页面中出现了，但是跟踪后台报错：
```shell script
Cannot feed value of shape (1, 416, 416, 3) for Tensor 'input_1:0', which has shape '(?, 608, 608, 3)'
KeyError: ('yolo', '0')
```
不能输入 416*416 的图像给模型，而是（下载后转化而城的模型）需要 608*608。

`yolo.py` 中改成：
```shell script
self.model_image_size = (608, 608) # (416, 416) 
```


* keras版本yolov3提示str object has no attribute decode

卸载原来的h5py模块，安装2.10版本
```shell script
pip install h5py==2.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

* cv2 读取[rtsp](https://www.jianshu.com/p/5d1e3c522e51)的一些弊端

 cv2.VideoCapture 的 read 函数并不能获取实时流的最新帧而是按照内部缓冲区中顺序逐帧的读取，opencv会每过一段时间清空一次缓冲区，但是清空的时机并不是我们能够控制的，因此如果对视频帧的处理速度如果跟不上接受速度，那么每过一段时间，在播放时(imshow)时会看到画面突然花屏，甚至程序直接崩溃，在网上查了很多资料，处理方式基本是一个思想

生产者消费者模式：
        使用一个临时缓存，可以是一个变量保存最新一帧，也可以是一个队列保存一些帧，然后开启一个线程读取最新帧保存到缓存里，用户读取的时候只返回最新的一帧
