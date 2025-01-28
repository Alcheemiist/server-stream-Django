# streaming/urls.py

from django.urls import path
from . import views
from .views import InferenceResultViewSet


urlpatterns = [

    path('upload-inference-results/', InferenceResultViewSet.as_view({'post': 'create'})),
    path('', views.index, name='index'),
    path('video_feed/<str:client_id>/', views.live_view, name='live_view'),
    path('mjpeg_stream/<str:client_id>/', views.mjpeg_stream, name='mjpeg_stream'),
    path('recording/<str:filename>/', views.download_recording, name='download_recording'),
    path('get_metrics/<str:client_id>/', views.get_metrics, name='get_metrics'),
    path('receive_json/', views.receive_json, name='receive_json'),

]
