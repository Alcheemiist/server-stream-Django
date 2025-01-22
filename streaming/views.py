# streaming/views.py

import os
import cv2
import numpy as np
from django.http import StreamingHttpResponse, HttpResponse, FileResponse, JsonResponse
from django.shortcuts import render
from .consumers import client_frames, client_ids, video_metadata, client_stats
from datetime import datetime
import json
import asyncio  # Import asyncio for async operations
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


async def stream_frames(client_id):
    while True:
        frame = client_frames.get(client_id, None)
        if frame is not None:
            # Rotate the frame by 90 degrees clockwise
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            success, buffer = cv2.imencode('.jpg', rotated_frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send a blank frame if no real data is available
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            success, buffer = cv2.imencode('.jpg', blank_frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.1)  # Adjust sleep time as needed


async def mjpeg_stream(request, client_id):
    """
    Asynchronous view that returns a StreamingHttpResponse with multipart/x-mixed-replace content.
    """
    response = StreamingHttpResponse(
        stream_frames(client_id),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    return response

def index(request):
    """
    Main dashboard:
      - Shows currently connected clients.
      - Shows archived recordings (with metadata).
    """
    # Sort archived videos by start_time descending
    all_recordings = sorted(
        video_metadata.values(),
        key=lambda x: x['start_time'],
        reverse=True
    )
    return render(request, 'index.html', {
        'current_clients': list(client_ids),
        'archived_recordings': all_recordings
    })

def live_view(request, client_id):
    """
    Live View page for <client_id>.
    Embeds the MJPEG stream in /mjpeg_stream/<client_id>
    and includes JavaScript to poll for metrics (FPS, bitrate, etc.).
    """
    return render(request, 'live_view.html', {'client_id': client_id})

def download_recording(request, filename):
    """
    Serve recorded video files from 'streaming_video' folder.
    """
    file_path = os.path.join('streaming_video', filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), content_type='video/avi')
    else:
        return HttpResponse("File not found.", status=404)

def get_metrics(request, client_id):
    """
    Returns JSON metrics for the given client_id.
    """
    if client_id not in client_stats:
        data = {
            "client_id": client_id,
            "status": "disconnected",
            "fps": 0.0,
            "resolution": "N/A",
            "duration": "0s",
            "bitrate": "0.00 kbps"
        }
        return HttpResponse(json.dumps(data), content_type='application/json')

    stats = client_stats[client_id]
    now = datetime.now()
    status = "connected" if (client_id in client_ids) else "disconnected"
    if stats["start_time"]:
        duration_sec = (now - stats["start_time"]).total_seconds()
    else:
        duration_sec = 0

    kbps = 0.0
    if duration_sec > 0:
        kbps = (stats["sum_of_bytes"] * 8) / 1024.0 / duration_sec

    data = {
        "client_id": client_id,
        "status": status,
        "fps": round(stats["fps"], 2),
        "resolution": f'{stats["width"]}x{stats["height"]}' if stats["width"] and stats["height"] else "N/A",
        "duration": f'{int(duration_sec)}s',
        "bitrate": f'{kbps:.2f} kbps'
    }
    return HttpResponse(json.dumps(data), content_type='application/json')


@csrf_exempt
def receive_json(request):
    """
    Receives JSON data via POST and returns a simple JSON response.
    Also broadcasts the data to connected WebSocket clients so they
    can update instantly.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Do something with 'data', e.g. store in DB or broadcast

            # (Optional) Broadcast via Channels to all listening websockets
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'json_updates',         # group name
                {
                    'type': 'new_json', # custom event type in consumer
                    'payload': data,     # the actual data
                }
            )

            return JsonResponse({'status': 'success', 'message': 'Data received'}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)