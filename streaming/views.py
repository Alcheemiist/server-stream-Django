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
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


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
    Receives JSON data via POST and returns a JSON response.
    This version supports both:
      - A raw JSON POST with content type 'application/json'
      - A multipart/form-data POST with a file field "json_file"
    The received JSON is also saved to disk.
    Additionally, the data is broadcasted to WebSocket clients via Django Channels.
    """
    if request.method == 'POST':
        try:
            # Determine if the client sent a file upload or raw JSON
            if request.content_type.startswith('multipart/form-data'):
                # Expecting a file field named "json_file"
                if 'json_file' not in request.FILES:
                    return JsonResponse({'status': 'error', 'message': 'No json_file provided'}, status=400)
                
                json_file = request.FILES['json_file']
                file_data = json_file.read().decode('utf-8')
                try:
                    data = json.loads(file_data)
                except json.JSONDecodeError as e:
                    return JsonResponse({'status': 'error', 'message': f'Invalid JSON in uploaded file: {str(e)}'}, status=400)
                # Save the received JSON file to a designated folder
                save_dir = os.path.join(settings.BASE_DIR, 'inference_results')
                os.makedirs(save_dir, exist_ok=True)
                filename = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path = os.path.join(save_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_data)
                logger.info(f"Saved uploaded JSON file to {file_path}")
            else:
                # Assume a raw JSON POST (content type application/json)
                try:
                    data = json.loads(request.body)
                except json.JSONDecodeError as e:
                    return JsonResponse({'status': 'error', 'message': f'Invalid JSON: {str(e)}'}, status=400)
                # Save the JSON content to a file as well
                save_dir = os.path.join(settings.BASE_DIR, 'inference_results')
                os.makedirs(save_dir, exist_ok=True)
                filename = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                file_path = os.path.join(save_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved raw JSON to {file_path}")

            # (Optional) Broadcast the JSON data via Django Channels to WebSocket clients
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'json_updates',         # Group name
                {
                    'type': 'new_json',  # Custom event type handled by your consumer
                    'payload': data,     # The JSON data payload
                }
            )

            return JsonResponse({'status': 'success', 'message': 'Data received and saved'}, status=200)
        except Exception as e:
            logger.error(f"Unexpected error in receive_json: {str(e)}")
            return JsonResponse({'status': 'error', 'message': f'Unexpected error: {str(e)}'}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST method allowed'}, status=405)
