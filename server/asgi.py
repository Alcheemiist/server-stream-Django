import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from streaming.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
django.setup()

application = ProtocolTypeRouter({
    # HTTP requests are handled by Django’s ASGI application
    "http": get_asgi_application(),
    # WebSocket requests are handled via Channels
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})
