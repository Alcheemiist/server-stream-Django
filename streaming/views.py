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



logger = logging.getLogger(__name__)




from rest_framework.viewsets import ViewSet


# Variable globale pour stocker les résultats d'inférence
INFERENCE_RESULTS = []


class InferenceResultViewSet(ViewSet):
    def create(self, request):
        """
        Traite une requête POST contenant un fichier JSON :
        1. Valide et nettoie les données JSON.
        2. Vérifie uniquement la résolution de la première entrée.
        3. Ajuste les bounding boxes si nécessaire.
        4. Stocke les résultats dans la variable globale INFERENCE_RESULTS.
        """
        global INFERENCE_RESULTS  # Permet d'accéder et de modifier la variable globale

        try:
            # Récupérer le fichier JSON téléversé
            json_file = request.FILES.get("json_file")
            if not json_file:
                return JsonResponse({"error": "Le fichier JSON est manquant."}, status=400)

            # Charger les données JSON
            try:
                data = json.load(json_file)
            except json.JSONDecodeError as e:
                return JsonResponse({"error": f"Le fichier JSON est invalide : {str(e)}"}, status=400)

            if not data or not isinstance(data, list):
                return JsonResponse({"error": "Le fichier JSON doit contenir un tableau d'objets."}, status=400)

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
            if width == 640 and height == 240:
                scaling_factor_x = 1.0
                scaling_factor_y = 1.0
            elif width == 1280 and height == 480:
                scaling_factor_x = 640 / 1280
                scaling_factor_y = 240 / 480
            elif width == 640 and height == 480:
                scaling_factor_x = 1.0
                scaling_factor_y = 240 / 480
            else:
                return JsonResponse(
                    {"error": f"Résolution non prise en charge : {width}x{height}."},
                    status=400,
                )

            # Nettoyage et traitement des données
            frame_counter = 0
            cleaned_data = []  # Stocke les données nettoyées pour la heatmap
            for inference_data in data:
                # Nettoyer et convertir le champ inference_time
                inference_time = inference_data.get("inference_time", "0")
                inference_time = inference_time.replace("ms", "").replace("s", "").strip()
                try:
                    inference_time = float(inference_time)
                    inference_data["inference_time"] = inference_time
                except ValueError:
                    return JsonResponse(
                        {"error": f"Le champ inference_time contient une valeur invalide : {inference_time}"},
                        status=400,
                    )

                # Générer un frame_id si absent
                if "frame_id" not in inference_data:
                    frame_counter += 1
                    inference_data["frame_id"] = f"frame_{frame_counter}"

                # Nettoyer et valider le champ timestamp
                try:
                    if isinstance(inference_data["timestamp"], int):
                        inference_data["timestamp"] = datetime.fromtimestamp(
                            inference_data["timestamp"] / 1000
                        ).isoformat()
                    elif isinstance(inference_data["timestamp"], str):
                        datetime.fromisoformat(inference_data["timestamp"])
                except (ValueError, KeyError, TypeError):
                    return JsonResponse(
                        {"error": f"Le champ 'timestamp' doit être une chaîne ISO 8601 valide ou un timestamp en millisecondes. Valeur actuelle : {inference_data.get('timestamp')}"},
                        status=400,
                    )

                # Validation des champs "bounding_box" et autres champs requis
                for i, detection in enumerate(inference_data.get("detections", [])):
                    bounding_box = detection.get("bounding_box")
                    if not bounding_box:
                        return JsonResponse(
                            {"error": f"La détection {i} ne contient pas de champ 'bounding_box'."},
                            status=400,
                        )
                    missing_keys = [key for key in ["x_min", "y_min", "x_max", "y_max"] if key not in bounding_box]
                    if missing_keys:
                        return JsonResponse(
                            {"error": f"Les champs suivants sont manquants dans 'bounding_box' de la détection {i} : {', '.join(missing_keys)}."},
                            status=400,
                        )

                    # Ajuster les bounding boxes si nécessaire
                    try:
                        bounding_box["x_min"] = int(bounding_box["x_min"] * scaling_factor_x)
                        bounding_box["y_min"] = int(bounding_box["y_min"] * scaling_factor_y)
                        bounding_box["x_max"] = int(bounding_box["x_max"] * scaling_factor_x)
                        bounding_box["y_max"] = int(bounding_box["y_max"] * scaling_factor_y)
                        detection["confidence"] = float(detection["confidence"])
                        detection["area"] = float(detection.get("area", 0))
                        detection["center"] = {
                            "x": int(detection["center"]["x"] * scaling_factor_x),
                            "y": int(detection["center"]["y"] * scaling_factor_y),
                        }
                    except (ValueError, KeyError) as e:
                        return JsonResponse(
                            {"error": f"Erreur de conversion des valeurs pour une détection : {str(e)}"},
                            status=400,
                        )

                # Ajouter les données nettoyées à la liste
                cleaned_data.append(inference_data)

            # Stocker les données nettoyées dans la variable globale
            INFERENCE_RESULTS = cleaned_data
            logger.info(f"Les données d'inférence ont été mises à jour : {len(INFERENCE_RESULTS)} résultats stockés.")

            # Appeler les fonctions save_data_to_main_db et filter_and_transfer_data
            """
            save_data_to_main_db(cleaned_data)
            for result in InferenceResult.objects.all():
                filter_and_transfer_data(result)
                """

            # Si toutes les validations passent, renvoyer une réponse de succès
            return JsonResponse(
                {"status": "success", "message": "Les données JSON ont été traitées ."},
                status=200,
            )

        except Exception as e:
            logger.error(f"Une erreur inattendue est survenue : {str(e)}")
            return JsonResponse({"error": f"Une erreur inattendue est survenue : {str(e)}"}, status=500)
            

async def stream_frames(client_id, enable_heatmap=False):
    """
    Génère un flux MJPEG où chaque frame de stream (1 FPS) inclut une heatmap
    basée sur 10 frames d'inférence (10 FPS).
    """
    global INFERENCE_RESULTS  # Accès aux données globales

    # Configuration de la heatmap
    cell_size = 40
    alpha = 0.4
    decay_factor = 0.95
    frame_width, frame_height = 640, 480
    n_cols = frame_width // cell_size
    n_rows = frame_height // cell_size
    heat_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)

    frame_index = 0  # Index pour parcourir les frames d'inférence

    while True:
        # Récupérer la frame de stream actuelle
        frame = client_frames.get(client_id, None)

        if frame is not None:
            frame = cv2.resize(frame, (frame_width, frame_height))  # Redimensionner la frame

            # Si la heatmap est activée et qu'il y a des résultats
            if enable_heatmap and INFERENCE_RESULTS:
                heat_matrix *= decay_factor  # Appliquer la décroissance de la heatmap

                # Parcourir les 10 frames d'inférence et mettre à jour la heatmap
                for _ in range(10):
                    if frame_index < len(INFERENCE_RESULTS):
                        current_data = INFERENCE_RESULTS[frame_index]
                        for detection in current_data.get("detections", []):
                            if detection.get("confidence", 0) > 0.5:  # Seuil de confiance
                                bbox = detection.get("bounding_box", {})
                                if "x_min" in bbox and "y_min" in bbox and "x_max" in bbox and "y_max" in bbox:
                                    x_min, y_min = bbox["x_min"] // 2, bbox["y_min"] // 2
                                    x_max, y_max = bbox["x_max"] // 2, bbox["y_max"] // 2
                                    r_min, c_min = int(y_min // cell_size), int(x_min // cell_size)
                                    r_max, c_max = int(y_max // cell_size), int(x_max // cell_size)
                                    r_min, r_max = max(0, r_min), min(n_rows - 1, r_max)
                                    c_min, c_max = max(0, c_min), min(n_cols - 1, c_max)
                                    heat_matrix[r_min:r_max + 1, c_min:c_max + 1] += 1

                        # Passer à la prochaine frame d'inférence
                        frame_index += 1
                    else:
                        break

                # Appliquer la heatmap sur la frame
                temp_heat_matrix = cv2.resize(heat_matrix, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                if np.max(temp_heat_matrix) > 0:
                    temp_heat_matrix = (temp_heat_matrix / np.max(temp_heat_matrix) * 255).astype(np.uint8)
                else:
                    temp_heat_matrix = np.zeros_like(temp_heat_matrix, dtype=np.uint8)
                image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 1 - alpha, image_heat, alpha, 0)

            # Rotation de la frame
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Encoder la frame en JPEG
            success, buffer = cv2.imencode('.jpg', rotated_frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Frame vide si aucune donnée
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            success, buffer = cv2.imencode('.jpg', blank_frame)
            if success:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Attente de 1 seconde pour limiter le flux à 1 FPS
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