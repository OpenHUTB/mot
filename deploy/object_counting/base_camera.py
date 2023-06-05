import time
import threading
import imagezmq

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent:
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


# 各种跟踪方法的相机处理类的基类（camera_yolo.py）
class BaseCamera:
    threads = {}        # 从摄像头读图像帧的后台线程
    frame = {}          # 使用后台线程存储在这里的当前帧
    last_access = {}    # 最后一个客户端访问相机的时间（起到如果没有人访问就暂停服务的作用）
    event = {}

    def __init__(self, feed_type, device, port_list):
        """如果还没有运行就开始后端相机线程"""
        self.unique_name = (feed_type, device)
        BaseCamera.event[self.unique_name] = CameraEvent()

        if self.unique_name not in BaseCamera.threads:
            BaseCamera.threads[self.unique_name] = None
        if BaseCamera.threads[self.unique_name] is None:
            BaseCamera.last_access[self.unique_name] = time.time()

            # 启动后台处理图像帧的线程
            BaseCamera.threads[self.unique_name] = threading.Thread(target=self._thread,
                                                                    args=(self.unique_name, port_list))
            BaseCamera.threads[self.unique_name].start()

            # 进行等待，直到获得图像帧
            while self.get_frame(self.unique_name) is None:
                time.sleep(0)

    @classmethod
    def get_frame(cls, unique_name):
        """返回当前相机帧"""
        BaseCamera.last_access[unique_name] = time.time()

        # 等待从相机线程来的信号
        BaseCamera.event[unique_name].wait()
        BaseCamera.event[unique_name].clear()

        return BaseCamera.frame[unique_name]

    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @staticmethod
    def yolo_frames(unique_name):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @staticmethod
    def server_frames(image_hub):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def yolo_thread(cls, unique_name):
        device = unique_name[1]

        frames_iterator = cls.yolo_frames(unique_name)
        try:
            for frame in frames_iterator:
                BaseCamera.frame[unique_name] = frame
                BaseCamera.event[unique_name].set()  # send signal to clients
                time.sleep(0)
                if time.time() - BaseCamera.last_access[unique_name] > 60:
                    frames_iterator.close()
                    print('Stopping YOLO thread for device {} due to inactivity.'.format(device))
                    pass
        except Exception as e:
            BaseCamera.event[unique_name].set()  # send signal to clients
            frames_iterator.close()
            print('Stopping YOLO thread for device {} due to error.'.format(device))
            print(e)

    @classmethod
    def server_thread(cls, unique_name, port):
        device = unique_name[1]

        image_hub = imagezmq.ImageHub(open_port='tcp://*:{}'.format(port))

        frames_iterator = cls.server_frames(image_hub)
        try:
            for cam_id, frame in frames_iterator:
                BaseCamera.frame[unique_name] = cam_id, frame
                BaseCamera.event[unique_name].set()  # send signal to clients
                time.sleep(0)
                if time.time() - BaseCamera.last_access[unique_name] > 5:
                    frames_iterator.close()
                    image_hub.zmq_socket.close()
                    print('Closing server socket at port {}.'.format(port))
                    print('Stopping server thread for device {} due to inactivity.'.format(device))
                    pass
        except Exception as e:
            frames_iterator.close()
            image_hub.zmq_socket.close()
            print('Closing server socket at port {}.'.format(port))
            print('Stopping server thread for device {} due to error.'.format(device))
            print(e)

    @classmethod
    def _thread(cls, unique_name, port_list):
        feed_type, device = unique_name
        if feed_type == 'camera':
            port = port_list[int(device)]
            print('Starting server thread for device {} at port {}.'.format(device, port))
            cls.server_thread(unique_name, port)

        elif feed_type == 'yolo':
            """Camera background thread."""
            print('Starting YOLO thread for device {}.'.format(device))
            cls.yolo_thread(unique_name)

        BaseCamera.threads[unique_name] = None
