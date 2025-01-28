from django.db import models


class ImageSize(models.Model):
    """
    Dimensions de l'image d'entrée.
    """
    width = models.IntegerField()  # Largeur de l'image en pixels
    height = models.IntegerField()  # Hauteur de l'image en pixels

    # Suppression de la contrainte d'unicité
    class Meta:
        pass

    def __str__(self):
        return f"{self.width}x{self.height}"


class BoundingBox(models.Model):
    """
    Coordonnées de la boîte englobante.
    """
    x_min = models.IntegerField()  # Coordonnée X minimale
    y_min = models.IntegerField()  # Coordonnée Y minimale
    x_max = models.IntegerField()  # Coordonnée X maximale
    y_max = models.IntegerField()  # Coordonnée Y maximale

    class Meta:
        pass

    def __str__(self):
        return f"({self.x_min}, {self.y_min}) - ({self.x_max}, {self.y_max})"


class Center(models.Model):
    """
    Coordonnées du centre de l'objet détecté.
    """
    x = models.IntegerField()  # Coordonnée X du centre
    y = models.IntegerField()  # Coordonnée Y du centre

    class Meta:
        pass

    def __str__(self):
        return f"({self.x}, {self.y})"


class InferenceResult(models.Model):
    """
    Résultat d'une inférence, contenant des détections.
    """
    timestamp = models.DateTimeField()  # Horodatage de l'inférence
    inference_time = models.FloatField()  # Temps de traitement (en secondes)
    frame_id = models.CharField(max_length=100)  # Identifiant de la frame
    model = models.CharField(max_length=100)  # Modèle utilisé
    device = models.CharField(max_length=100)  # Appareil utilisé
    zone = models.CharField(max_length=100)  # Zone de détection
    image_size = models.ForeignKey(ImageSize, on_delete=models.CASCADE)  # Dimensions de l'image

    def __str__(self):
        return f"InferenceResult {self.id} at {self.timestamp}"


class Detection(models.Model):
    """
    Objet détecté dans une image.
    """
    CLASS_CHOICES = [
        ('person', 'Person'),
        ('car', 'Car'),
        ('bus', 'Bus'),
        ('truck', 'Truck'),
        ('empty garbage', 'Empty Garbage'),
        ('full garbage', 'Full Garbage'),
        # Ajoutez d'autres classes si nécessaire
    ]

    inference_result = models.ForeignKey(
        InferenceResult, related_name='detections', on_delete=models.CASCADE, null=True, blank=True
    )
    class_id = models.IntegerField()  # ID de la classe
    class_name = models.CharField(max_length=50, choices=CLASS_CHOICES)  # Nom de la classe
    confidence = models.FloatField()  # Confiance dans la détection
    bounding_box = models.ForeignKey(BoundingBox, on_delete=models.CASCADE)  # Boîte englobante
    area = models.FloatField(default=0.0)  # Aire de la boîte
    center = models.ForeignKey(Center, on_delete=models.CASCADE)  # Coordonnées du centre

    def __str__(self):
        return f"{self.class_name} ({self.confidence * 100:.1f}%)"
