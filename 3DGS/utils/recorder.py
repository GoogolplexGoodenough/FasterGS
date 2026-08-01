import cv2
import os
import threading
import queue
from collections import defaultdict
import numpy as np
import time


class AsyncVideoRecorder:
    def __init__(self, output_dir, fps=30, frame_size=(640, 480), codec='mp4v'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.fps = fps
        self.frame_size = frame_size
        self.codec = cv2.VideoWriter_fourcc(*codec)

        self.frame_queues = defaultdict(queue.Queue)
        self.video_writers = {}
        self.worker_threads = {}
        self.stop_flags = {}
        self.lock = threading.Lock()

    def _init_writer(self, name):
        path = os.path.join(self.output_dir, f"{name}.mp4")
        writer = cv2.VideoWriter(path, self.codec, self.fps, self.frame_size)
        self.video_writers[name] = writer

    def _resize_if_needed(self, frame):
        if frame.shape[:2][::-1] != self.frame_size:
            return cv2.resize(frame, self.frame_size)
        return frame

    def _worker(self, name):
        q = self.frame_queues[name]
        while True:
            frame = q.get()
            # if frame is None:  # 停止信号
            #     break
            if frame is None:
                time.sleep(0.01)
                continue
            if isinstance(frame, str):
                break
            frame = self._resize_if_needed(frame)
            with self.lock:
                if name not in self.video_writers:
                    self._init_writer(name)
                if not isinstance(frame, np.uint8):
                    frame = (frame.clip(0, 1) * 255).astype(np.uint8)
                self.video_writers[name].write(frame)
        # 收尾
        with self.lock:
            self.video_writers[name].release()
            print(f"[{name}] video released.")

    def put_frame(self, name, frame):
        if name not in self.worker_threads:
            t = threading.Thread(target=self._worker, args=(name,), daemon=True)
            self.worker_threads[name] = t
            t.start()
        self.frame_queues[name].put(frame)

    def finish(self):
        for name, q in self.frame_queues.items():
            q.put("end")  # 发送停止信号
        for name, t in self.worker_threads.items():
            t.join()
        print("All videos written and saved.")
