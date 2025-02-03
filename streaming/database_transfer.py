from django.db import connections
from .models import InferenceResult, Detection, BoundingBox, Center, ImageSize, DeviceZone

CATEGORY_MAPPING = {
    'people': ['person'],
    'vehicles': ['car', 'taxi', 'bus', 'truck'],
    'garbage': ['empty garbage', 'full garbage', 'garbage truck']
}

def synchronize_sequences():
    """
    Synchronise les séquences PostgreSQL avec les valeurs actuelles des tables dans toutes les bases de données.
    """
    databases = ['default', 'people', 'vehicles', 'garbage']  # Toutes les bases secondaires
    tables_with_sequences = [
        'streaming_inferenceresult',
        'streaming_imagesize',
        'streaming_boundingbox',
        'streaming_center',
        'streaming_detection',
    ]

    for db_name in databases:
        try:
            with connections[db_name].cursor() as cursor:
                for table in tables_with_sequences:
                    cursor.execute(f"""
                        SELECT setval(
                            pg_get_serial_sequence('{table}', 'id'),
                            COALESCE((SELECT MAX(id) FROM {table}), 1) + 1,
                            false
                        );
                    """)
        except Exception as e:
            raise Exception(f"Erreur lors de la synchronisation des séquences pour la base '{db_name}': {str(e)}")

def filter_and_transfer_data(inference_result):
    """
    Filtre et transfère les données vers les bases secondaires en respectant l'ordre d'insertion.
    """
    for db_name, categories in CATEGORY_MAPPING.items():
        try:
            filtered_detections = inference_result.detections.filter(class_name__in=categories)

            if filtered_detections.exists():
                with connections[db_name].cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO streaming_imagesize (width, height)
                        VALUES (%s, %s)
                        RETURNING id
                        """,
                        [
                            inference_result.image_size.width,
                            inference_result.image_size.height
                        ]
                    )
                    image_size_id = cursor.fetchone()[0]
                    synchronize_sequences()

                    cursor.execute(
                        """
                        INSERT INTO streaming_inferenceresult (timestamp, inference_time, frame_id, model, device, image_size_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        [
                            inference_result.timestamp,
                            inference_result.inference_time,
                            inference_result.frame_id,
                            inference_result.model,
                            inference_result.device,
                            image_size_id
                        ]
                    )
                    new_inference_id = cursor.fetchone()[0]
                    synchronize_sequences()

                    for det in filtered_detections:
                        cursor.execute(
                            """
                            INSERT INTO streaming_boundingbox (x_min, y_min, x_max, y_max)
                            VALUES (%s, %s, %s, %s)
                            RETURNING id
                            """,
                            [
                                det.bounding_box.x_min,
                                det.bounding_box.y_min,
                                det.bounding_box.x_max,
                                det.bounding_box.y_max
                            ]
                        )
                        bounding_box_id = cursor.fetchone()[0]
                        synchronize_sequences()

                        cursor.execute(
                            """
                            INSERT INTO streaming_center (x, y)
                            VALUES (%s, %s)
                            RETURNING id
                            """,
                            [
                                det.center.x,
                                det.center.y
                            ]
                        )
                        center_id = cursor.fetchone()[0]
                        synchronize_sequences()

                        cursor.execute(
                            """
                            INSERT INTO streaming_detection (class_id, class_name, confidence, bounding_box_id, area, center_id, inference_result_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            [
                                det.class_id,
                                det.class_name,
                                det.confidence,
                                bounding_box_id,
                                det.area,
                                center_id,
                                new_inference_id
                            ]
                        )
                        synchronize_sequences()

        except Exception as e:
            print(f"Erreur avec la base de données '{db_name}': {str(e)}")
