# 摄像头客户端
from imutils.video import VideoStream
import imagezmq

# 打开远端的相机
path = "rtsp://d:jjj@10.69.197.38:8554/live"
# path = "rtsp://192.168.1.69:8080//h264_ulaw.sdp"  # change to your IP stream address
cap = VideoStream(path)

# 将摄像头的射频流发送给服务器端处理
sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')  # 修改为服务器线程的 IP 地址和端口
# 左上角显示的每个摄像头流的名称
cam_id = 'Camera 1'  # 中文乱码 Camera 1; this name will be displayed on the corresponding camera stream

stream = cap.start()

while True:
    frame = stream.read()
    sender.send_image(cam_id, frame)
