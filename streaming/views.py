# streaming/views.py

import os
import cv2
from django.conf import settings
from .save_data_service import save_data_to_main_db
from .database_transfer import filter_and_transfer_data
from .models import InferenceResult
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
import os
import cv2
import numpy as np
from django.http import StreamingHttpResponse, HttpResponse, FileResponse, JsonResponse
from django.shortcuts import render
from .consumers import client_frames, client_ids, video_metadata, client_stats, VideoStreamConsumer
from datetime import datetime
import json
import asyncio
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import logging
from django.http import JsonResponse



logger = logging.getLogger(__name__)




from rest_framework.viewsets import ViewSet


# Variable globale pour stocker les résultats d'inférence


# Variable globale pour stocker les résultats d'inférence
INFERENCE_RESULTS = []
logger = logging.getLogger(__name__)

class InferenceResultViewSet(ViewSet):
    def create(self, request):
        """
        Traite une requête POST contenant du JSON brut :
        1. Valide et nettoie les données JSON.
        2. Vérifie uniquement la résolution de la première entrée.
        3. Ajuste les bounding boxes si nécessaire.
        4. Stocke les résultats dans la variable globale INFERENCE_RESULTS.
        """
        global INFERENCE_RESULTS  # Permet d'accéder et de modifier la variable globale

        try:
            # Charger les données JSON depuis le corps de la requête
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                return JsonResponse({"error": f"Le JSON est invalide : {str(e)}"}, status=400)

            # Vérifier que le JSON est un tableau d'objets
            if not data or not isinstance(data, list):
                return JsonResponse({"error": "Le JSON doit contenir un tableau d'objets."}, status=400)

            # Vérification de la résolution à partir de la première entrée
            first_entry = data[0]
            image_size = first_entry.get("image_size", {})
            width = image_size.get("width")
            height = image_size.get("height")

            if width is None or height is None:
                return JsonResponse(
                    {"error": "Les dimensions de l'image sont manquantes dans la première entrée."},
                    status=400,
                )

            # Déterminer les facteurs d'échelle pour les résolutions prises en charge
            scaling_factors = {
                (640, 240): (1.0, 1.0),
                (1280, 480): (640 / 1280, 240 / 480),
                (640, 480): (1.0, 240 / 480)
            }
            scaling_factor_x, scaling_factor_y = scaling_factors.get((width, height), (None, None))

            if scaling_factor_x is None or scaling_factor_y is None:
                return JsonResponse(
                    {"error": f"Résolution non prise en charge : {width}x{height}."},
                    status=400,
                )

            # Nettoyage et traitement des données
            frame_counter = 0
            cleaned_data = []  
            for inference_data in data:
                # Nettoyage du champ inference_time
                inference_time = inference_data.get("inference_time", "0").replace("ms", "").replace("s", "").strip()
                try:
                    inference_data["inference_time"] = float(inference_time)
                except ValueError:
                    return JsonResponse({"error": f"Valeur invalide pour 'inference_time': {inference_time}"}, status=400)

                # Générer un frame_id si absent
                if "frame_id" not in inference_data:
                    frame_counter += 1
                    inference_data["frame_id"] = f"frame_{frame_counter}"

                # Validation et conversion du timestamp
                try:
                    if isinstance(inference_data["timestamp"], int):
                        inference_data["timestamp"] = datetime.fromtimestamp(inference_data["timestamp"] / 1000).isoformat()
                    elif isinstance(inference_data["timestamp"], str):
                        datetime.fromisoformat(inference_data["timestamp"])
                except (ValueError, KeyError, TypeError):
                    return JsonResponse({"error": f"Timestamp invalide : {inference_data.get('timestamp')}"}, status=400)

                # Vérification et ajustement des bounding boxes
                for i, detection in enumerate(inference_data.get("detections", [])):
                    bounding_box = detection.get("bounding_box")
                    if not bounding_box:
                        return JsonResponse({"error": f"Détection {i} sans 'bounding_box'."}, status=400)

                    missing_keys = [key for key in ["x_min", "y_min", "x_max", "y_max"] if key not in bounding_box]
                    if missing_keys:
                        return JsonResponse(
                            {"error": f"Champs manquants dans 'bounding_box' {i} : {', '.join(missing_keys)}."},
                            status=400,
                        )

                    # Ajustement des valeurs de bounding box
                    try:
                        for key in ["x_min", "y_min", "x_max", "y_max"]:
                            bounding_box[key] = int(bounding_box[key] * (scaling_factor_x if "x" in key else scaling_factor_y))
                        detection["confidence"] = float(detection["confidence"])
                        detection["area"] = float(detection.get("area", 0))
                    except (ValueError, KeyError) as e:
                        return JsonResponse({"error": f"Erreur dans les valeurs de bounding box : {str(e)}"}, status=400)

                cleaned_data.append(inference_data)

            # Stocker les données nettoyées
            INFERENCE_RESULTS = cleaned_data
            logger.info(f"{len(INFERENCE_RESULTS)} résultats stockés.")

            return JsonResponse({"status": "success", "message": "Données JSON traitées."}, status=200)
            """
            save_data_to_main_db(cleaned_data)
            for result in InferenceResult.objects.all():
                filter_and_transfer_data(result)
                """
        except Exception as e:
            logger.error(f"Erreur inattendue : {str(e)}")
            return JsonResponse({"error": f"Erreur inattendue : {str(e)}"}, status=500)


async def stream_frames(client_id, enable_heatmap=False):
    """
    Génère un flux MJPEG où chaque frame inclut une heatmap.
    """
    global INFERENCE_RESULTS  

    cell_size = 40
    alpha = 0.4
    frame_width, frame_height = 640, 480
    n_cols, n_rows = frame_width // cell_size, frame_height // cell_size
    heat_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)

    frame_index = 0  

    while True:
        frame = client_frames.get(client_id, None)

        if frame is not None:
            frame = cv2.resize(frame, (frame_width, frame_height))

            if enable_heatmap and INFERENCE_RESULTS:
                heat_matrix *= 0.95  
                for _ in range(10):
                    if frame_index < len(INFERENCE_RESULTS):
                        current_data = INFERENCE_RESULTS[frame_index]
                        for detection in current_data.get("detections", []):
                            if detection.get("confidence", 0) > 0.5:
                                bbox = detection.get("bounding_box", {})
                                if all(k in bbox for k in ["x_min", "y_min", "x_max", "y_max"]):
                                    r_min, c_min = int(bbox["y_min"] // cell_size), int(bbox["x_min"] // cell_size)
                                    r_max, c_max = int(bbox["y_max"] // cell_size), int(bbox["x_max"] // cell_size)
                                    heat_matrix[r_min:r_max + 1, c_min:c_max + 1] += 1
                        frame_index += 1
                    else:
                        break

                temp_heat_matrix = cv2.resize(heat_matrix, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                temp_heat_matrix = (temp_heat_matrix / np.max(temp_heat_matrix) * 255).astype(np.uint8) if np.max(temp_heat_matrix) > 0 else np.zeros_like(temp_heat_matrix)
                frame = cv2.addWeighted(frame, 1 - alpha, cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET), alpha, 0)

            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        await asyncio.sleep(1)

# Vue MJPEG existante (avec option heatmap)
async def mjpeg_stream(request, client_id):
    enable_heatmap = request.GET.get('heatmap', 'false').lower() == 'true'
    response = StreamingHttpResponse(
        stream_frames(client_id, enable_heatmap=enable_heatmap),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
    return response

# Vue Live View existante (avec option heatmap)
def live_view(request, client_id):
    enable_heatmap = request.GET.get('heatmap', 'false').lower() == 'true'
    return render(request, 'live_view.html', {
        'client_id': client_id,
        'enable_heatmap': enable_heatmap
    })








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