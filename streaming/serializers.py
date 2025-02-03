# api/serializers.py


from rest_framework import serializers
from .models import InferenceResult, Detection, ImageSize, BoundingBox, Center

class ImageSizeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageSize
        fields = ['width', 'height']

class BoundingBoxSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoundingBox
        fields = ['x_min', 'y_min', 'x_max', 'y_max']

class CenterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Center
        fields = ['x', 'y']

class DetectionSerializer(serializers.ModelSerializer):
    bounding_box = BoundingBoxSerializer()
    center = CenterSerializer()

    class Meta:
        model = Detection
        fields = ['class_id', 'class_name', 'confidence', 'bounding_box', 'area', 'center']

class InferenceResultSerializer(serializers.ModelSerializer):
    image_size = ImageSizeSerializer()
    detections = DetectionSerializer(many=True)

    class Meta:
        model = InferenceResult
        fields = ['timestamp', 'inference_time', 'frame_id', 'model', 'device', 'zone', 'image_size', 'detections']

class InferenceResultListSerializer(serializers.ListSerializer):
    child = InferenceResultSerializer()

    def create(self, validated_data):
        inference_results = []
        for item in validated_data:
            image_size_data = item.pop('image_size')
            detections_data = item.pop('detections')

            image_size = ImageSize.objects.create(**image_size_data)
            inference_result = InferenceResult.objects.create(image_size=image_size, **item)
            inference_results.append(inference_result)

            detection_instances = []
            for detection in detections_data:
                bbox_data = detection.pop('bounding_box')
                center_data = detection.pop('center')
                bbox = BoundingBox.objects.create(**bbox_data)
                center = Center.objects.create(**center_data)
                detection_instance = Detection(
                    inference_result=inference_result,
                    bounding_box=bbox,
                    center=center,
                    **detection
                )
                detection_instances.append(detection_instance)
            Detection.objects.bulk_create(detection_instances)
        return inference_results

class BulkInferenceResultSerializer(InferenceResultSerializer):
    class Meta(InferenceResultSerializer.Meta):
        list_serializer_class = InferenceResultListSerializer
