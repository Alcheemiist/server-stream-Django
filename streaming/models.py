from django.db import models

# Base principale (default)
class DeviceZone(models.Model):
    device_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=100)
    width = models.IntegerField()
    height = models.IntegerField()
    last_active = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.device_id} - {self.name}"

class ZoneType(models.Model):
    ZONE_TYPES = [
        ('parking', 'Parking'),
        ('garbage_truck', 'Garbage Truck Zone')
    ]
    device_zone = models.ForeignKey(DeviceZone, on_delete=models.CASCADE)
    zone_type = models.CharField(max_length=50, choices=ZONE_TYPES)
    bounding_box = models.JSONField()

    def __str__(self):
        return f"{self.device_zone.name} - {self.zone_type}"

class ImageSize(models.Model):
    width = models.IntegerField()
    height = models.IntegerField()

    def __str__(self):
        return f"{self.width}x{self.height}"

class BoundingBox(models.Model):
    x_min = models.IntegerField()
    y_min = models.IntegerField()
    x_max = models.IntegerField()
    y_max = models.IntegerField()

    def __str__(self):
        return f"({self.x_min}, {self.y_min}) - ({self.x_max}, {self.y_max})"

class Center(models.Model):
    x = models.IntegerField()
    y = models.IntegerField()

    def __str__(self):
        return f"({self.x}, {self.y})"

class InferenceResult(models.Model):
    timestamp = models.DateTimeField()
    inference_time = models.FloatField()
    frame_id = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    device = models.CharField(max_length=100)
    image_size = models.ForeignKey(ImageSize, on_delete=models.CASCADE)

    def get_device_zone(self):
        return DeviceZone.objects.filter(device_id=self.device).first()

class Detection(models.Model):
    CLASS_CHOICES = [
        ('person', 'Person'),
        ('car', 'Car'),
        ('taxi', 'Taxi'),
        ('bus', 'Bus'),
        ('truck', 'Truck'),
        ('empty garbage', 'Empty Garbage'),
        ('full garbage', 'Full Garbage'),
        ('garbage truck', 'Garbage Truck')
    ]
    inference_result = models.ForeignKey(InferenceResult, related_name='detections', on_delete=models.CASCADE)
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=50, choices=CLASS_CHOICES)
    confidence = models.FloatField()
    bounding_box = models.ForeignKey(BoundingBox, on_delete=models.CASCADE)
    area = models.FloatField(default=0.0)
    center = models.ForeignKey(Center, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.class_name} ({self.confidence * 100:.1f}%)"

# Modèle pour la gestion des alertes dans les bases secondaires
class Alert(models.Model):
    ALERT_CATEGORIES = [
        ('vehicle', 'Vehicle'),
        ('people', 'People'),
        ('garbage', 'Garbage')
    ]
    category = models.CharField(max_length=50, choices=ALERT_CATEGORIES)
    reason = models.CharField(max_length=255)
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE)
    device_zone = models.ForeignKey(DeviceZone, on_delete=models.CASCADE)
    timestamp_start = models.DateTimeField(null=True, blank=True)
    timestamp_end = models.DateTimeField(null=True, blank=True)
    first_detected_at = models.DateTimeField()
    duration = models.FloatField(default=0.0)
    bounding_box_ref = models.JSONField()
    frame_count_missing = models.IntegerField(default=0)
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('active', 'Active'), ('resolved', 'Resolved')])

    def __str__(self):
        return f"Alert {self.category} - {self.reason} ({self.status})"
