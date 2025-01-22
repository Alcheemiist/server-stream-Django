from django.urls import re_path, path
from . import consumers
from .consumers import DataConsumer

websocket_urlpatterns = [
    re_path(r"ws/stream/$", consumers.VideoStreamConsumer.as_asgi()),
    path('ws/json-updates/', DataConsumer.as_asgi()),

]
