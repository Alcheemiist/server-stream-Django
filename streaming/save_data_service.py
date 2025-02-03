from .models import InferenceResult, Detection, BoundingBox, Center, ImageSize, DeviceZone
from django.db import transaction, connections

def log_database_usage(db_name):
    """
    Log la base de données en cours d'utilisation et capture les erreurs.
    """
    try:
        print(f"Connexion à la base de données : {db_name}")
        with connections[db_name].cursor() as cursor:
            cursor.execute("SELECT current_database();")
            db_name_result = cursor.fetchone()
            print(f"Base active : {db_name_result[0]}")
    except Exception as e:
        raise Exception(f"Erreur avec la base de données '{db_name}': {str(e)}")

def synchronize_sequences():
    """
    Synchronise les séquences PostgreSQL avec les valeurs actuelles des tables dans la base de données.
    """
    tables = [
        "streaming_inferenceresult",
        "streaming_imagesize",
        "streaming_boundingbox",
        "streaming_center",
        "streaming_detection",
    ]
    
    with connections['default'].cursor() as cursor:
        for table in tables:
            try:
                cursor.execute(f"""
                    SELECT setval(
                        pg_get_serial_sequence('{table}', 'id'),
                        COALESCE((SELECT MAX(id) FROM {table}), 1) + 1,
                        false
                    );
                """)
                print(f"Synchronized sequence for table: {table}")
            except Exception as e:
                print(f"Error synchronizing sequence for table {table}: {e}")

def save_data_to_main_db(json_data):
    """
    Enregistre les données JSON dans la base principale tout en respectant les dépendances des modèles.
    """
    try:
        with transaction.atomic():
            for inference in json_data:
                # Associer le device à une zone existante
                
                # Toujours créer une nouvelle instance ImageSize
                image_size_data = inference.get("image_size", {})
                if not image_size_data:
                    raise ValueError("Les données de taille d'image sont manquantes.")

                image_size = ImageSize.objects.create(
                    width=image_size_data["width"],
                    height=image_size_data["height"]
                )

                # Créer une nouvelle instance InferenceResult
                inference_result = InferenceResult.objects.create(
                    timestamp=inference["timestamp"],
                    inference_time=float(inference["inference_time"]),
                    frame_id=inference["frame_id"],
                    model=inference["model"],
                    device=inference["device"],
                    image_size=image_size  # Associer la nouvelle instance ImageSize
                )

                # Parcourir les détections et gérer les objets associés
                for det in inference.get("detections", []):
                    bounding_box_data = det.get("bounding_box")
                    if not bounding_box_data:
                        raise ValueError("Les données de la boîte englobante sont manquantes.")

                    bounding_box = BoundingBox.objects.create(
                        x_min=bounding_box_data["x_min"],
                        y_min=bounding_box_data["y_min"],
                        x_max=bounding_box_data["x_max"],
                        y_max=bounding_box_data["y_max"]
                    )

                    center_data = det.get("center")
                    if not center_data:
                        raise ValueError("Les données du centre sont manquantes.")

                    center = Center.objects.create(
                        x=center_data["x"],
                        y=center_data["y"]
                    )

                    Detection.objects.create(
                        inference_result=inference_result,
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bounding_box=bounding_box,
                        area=det.get("area", 0),
                        center=center
                    )

            print("Toutes les données ont été enregistrées avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des données dans la base principale : {str(e)}")
        raise
