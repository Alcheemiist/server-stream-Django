# streaming/consumers.py

import cv2
import numpy as np
import os
import time
import random
import string
from collections import defaultdict
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer, WebsocketConsumer
import json

# Ensure the directory for storing videos exists
os.makedirs("streaming_video", exist_ok=True)

# Global data structures
client_frames = defaultdict(lambda: None)  # Latest frame per client
client_ids = set()                         # Active client IDs
video_writers = {}                         # cv2.VideoWriter objects
video_metadata = {}                        # Metadata for each recording
client_stats = defaultdict(lambda: {
    "start_time": None,
    "frame_count": 0,
    "fps": 0.0,
    "width": 0,
    "height": 0,
    "sum_of_bytes": 0
})

global_stream_index = 0  # To differentiate streams

def process_frame(frame_data):
    """Decode raw bytes into an OpenCV BGR frame."""
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.client_id = ''.join(random.choices(string.digits, k=5))
        client_ids.add(self.client_id)
        print(f"[CONNECT] New client: {self.client_id}")

        # Initialize stats
        start_time = datetime.now()
        client_stats[self.client_id]["start_time"] = start_time
        client_stats[self.client_id]["frame_count"] = 0
        client_stats[self.client_id]["fps"] = 0.0
        client_stats[self.client_id]["sum_of_bytes"] = 0

        # Initialize VideoWriter as None
        self.writer = None
        self.video_path = None

        # For FPS calculation
        self.frame_accumulator = 0
        self.last_fps_time = time.time()

    async def disconnect(self, close_code):
        # Release VideoWriter if exists
        if self.client_id in video_writers:
            video_writers[self.client_id].release()
            del video_writers[self.client_id]
            print(f"[DISCONNECT] Finalized recording for {self.client_id} at {self.video_path}")

            # Update metadata
            if self.video_path:
                fname = os.path.basename(self.video_path)
                if fname in video_metadata:
                    video_metadata[fname]['end_time'] = datetime.now()
                    try:
                        size = os.path.getsize(self.video_path)
                        video_metadata[fname]['filesize'] = size
                    except OSError:
                        video_metadata[fname]['filesize'] = 0

        # Clean up client data
        if self.client_id in client_ids:
            client_ids.remove(self.client_id)
        if self.client_id in client_frames:
            del client_frames[self.client_id]
        if self.client_id in client_stats:
            del client_stats[self.client_id]

        print(f"[DISCONNECT] Client {self.client_id} removed.")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Update sum_of_bytes for bitrate
            client_stats[self.client_id]["sum_of_bytes"] += len(bytes_data)

            # Decode frame
            frame = process_frame(bytes_data)
            if frame is None:
                return

            # Update the latest frame for MJPEG streaming
            client_frames[self.client_id] = frame

            # Initialize VideoWriter if not already
            if self.writer is None:
                h, w, _ = frame.shape
                client_stats[self.client_id]["width"] = w
                client_stats[self.client_id]["height"] = h

                start_time = client_stats[self.client_id]["start_time"]
                timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
                random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
                filename = f"client_{self.client_id}_{timestamp_str}_{random_suffix}.avi"
                self.video_path = os.path.join("streaming_video", filename)

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(self.video_path, fourcc, 15.0, (w, h))
                video_writers[self.client_id] = self.writer

                # Save metadata
                video_metadata[filename] = {
                    'filename': filename,
                    'client_id': self.client_id,
                    'start_time': start_time,
                    'end_time': None,
                    'filesize': 0
                }
                print(f"[RECORD] Started recording {self.client_id} -> {self.video_path}")

            # Write frame to disk
            if self.writer:
                self.writer.write(frame)

            # Update FPS every second
            client_stats[self.client_id]["frame_count"] += 1
            self.frame_accumulator += 1
            now_time = time.time()
            if (now_time - self.last_fps_time) >= 1.0:
                fps = self.frame_accumulator / (now_time - self.last_fps_time)
                client_stats[self.client_id]["fps"] = fps
                self.frame_accumulator = 0
                self.last_fps_time = now_time


class DataConsumer(WebsocketConsumer):
    def connect(self):
        # Join group so we can receive broadcasted messages
        self.group_name = 'json_updates'
        self.channel_layer.group_add(self.group_name, self.channel_name)
        self.accept()

    def disconnect(self, close_code):
        # Leave group
        self.channel_layer.group_discard(self.group_name, self.channel_name)

    # Custom handler for the "new_json" event
    def new_json(self, event):
        data = event['payload']
        # Send data to WebSocket client
        self.send(text_data=json.dumps(data))
        