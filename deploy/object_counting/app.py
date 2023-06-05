from importlib import import_module
from flask import Flask, render_template, Response
import cv2
import time

import threading
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow.python.client import device_lib

# 显示可用计算设备
print(device_lib.list_local_devices())
import tensorflow as tf
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
# sudo ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so.10.1 /usr/lib/x86_64-linux-gnu/libcudart.so.10.1
# sudo cp /usr/local/cuda-10.1/lib64/libcudart.so.10.1 /usr/local/lib/ && sudo ldconfig

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera_stream, feed_type, device):
    """Video streaming generator function."""
    unique_name = (feed_type, device)

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()

        cam_id, frame = camera_stream.get_frame(unique_name)
        if frame is None:
            break

        num_frames += 1

        time_now = time.time()
        total_time += time_now - time_start
        fps = num_frames / total_time

        # write camera name
        cv2.putText(frame, cam_id, (int(20), int(20 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)

        if feed_type == 'yolo':
            cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
                        (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Remove this line for test camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 定义视频源的应用程序路由
@app.route('/video_feed/<feed_type>/<device>')
def video_feed(feed_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    # port_list包含为每个服务器分配的端口列表。
    # camera_client_0.py 是设备号 0，它使用列表中的第一个端口 (5555)。
    # camera_client_1.py 是设备号 1，它使用列表中的第二个端口 (5566)。
    # 基本上，设备号对应于 `port_list` 中的索引，因此它们需要相应地匹配。 
    port_list = (5555, 5566)  # 5577
    if feed_type == 'camera':
        camera_stream = import_module('camera_server').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # 从网页页面传过来的对摄像头流进行检测请求
    elif feed_type == 'yolo':
        camera_stream = import_module('camera_yolo').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port_list), feed_type=feed_type, device=device),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def launch_camera_client(name):
    time.sleep(1)  # 等待服务线程app.run启动
    os.system("python camera_client_0.py")  # for debug


if __name__ == '__main__':
    # t1 = threading.Thread(target=launch_camera_client, args=('0',))
    # t1.start()

    app.run(host='0.0.0.0', threaded=True)
