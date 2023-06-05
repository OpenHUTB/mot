import threading

import cv2
import errno
import imagezmq
import os
import socket
import time


def create_streamer(file, connect_to='tcp://127.0.0.1:5555', loop=True):
    """
    You can use this function to emulate an IP camera for the counting apps.
    或者使用 ffserver: http://euhat.com/wp/2019/01/25/%E5%B0%86%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%8F%98%E6%88%90rtsp%E6%B5%81%E6%9C%8D%E5%8A%A1/

    Parameters
    ----------
    file : str
        Path to the video file you want to stream.
    connect_to : str, optional
        Where the video shall be streamed to.
        The default is 'tcp://127.0.0.1:5555'.
    loop : bool, optional
        Whether the video shall be looped. The default is True.

    Returns
    -------
    None.

    """

    # check if file exists and open capture
    if os.path.isfile(file):
        cap = cv2.VideoCapture(file)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

    # prepare streaming
    sender = imagezmq.ImageSender(connect_to=connect_to)
    host_name = socket.gethostname()

    while True:
        ret, frame = cap.read()

        if ret:
            # if a frame was returned, send it
            sender.send_image(host_name, frame)
        else:
            # if no frame was returned, either restart or stop the stream
            if loop:
                cap = cv2.VideoCapture(file)
            else:
                break


# 为每个视频流广播创建一个新的线程
def launch_video_streamer(video_name, connect_link):
    # time.sleep(1)  # 等待服务线程app.run启动
    create_streamer(video_name, connect_link)


if __name__ == '__main__':
    car_video = "/data3/dong/data/mot/deploy/videos/cidi_car.mp4"  # 十字路口的晴天
    # car_video = "/data3/dong/data/mot/deploy/0325_pic/2022-03-25-16-24-51/video.mp4"  # 十字路口的雨天
    # car_video = "/data3/dong/data/mot/deploy/0325_pic/2022-03-25-16-29-22/video.mp4"  # 绕圈
    # car_video = "/data3/dong/data/mot/deploy/0325_pic/2022-03-25-16-31-21/video.mp4"  # 高速
    connect_to_car = "tcp://127.0.0.1:5555"
    t1 = threading.Thread(target=launch_video_streamer, args=(car_video, connect_to_car))
    t1.start()
    # streamer_dongfanghong = create_streamer(dongfanghong_video)

    xinkeyuan_video = "/data3/dong/data/mot/deploy/videos/cidi_person.mp4"
    connect_to_xinkeyuan = "tcp://127.0.0.1:5566"
    t2 = threading.Thread(target=launch_video_streamer, args=(xinkeyuan_video, connect_to_xinkeyuan))
    t2.start()
    # streamer_xinkeyuan = create_streamer(xinkeyuan_video, connect_to_xinkeyuan)

    # lushannanlu_video = "/data3/dong/data/mot/deploy/videos/lushannanlu.mp4"
    # connect_to_lushannanlu = "tcp://127.0.0.1:5577"
    # t3 = threading.Thread(target=launch_video_streamer, args=(lushannanlu_video, connect_to_lushannanlu))
    # t3.start()
    # streamer_lushannanlu = create_streamer(lushannanlu_video, connect_to_lushannanlu)

