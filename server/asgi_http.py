import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
import streaming.routing  # we'll create this

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
django.setup()

application = ProtocolTypeRouter({
    # Django’s HTTP handling
    "http": get_asgi_application(),
   
})
