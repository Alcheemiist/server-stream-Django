# runservers.py

import os
import django
import asyncio
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

# Set up Django only once here.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
django.setup()

async def run_http_server():
    """
    Configures and starts the HTTP server on port 8000.
    """
    config = Config("server.asgi_http:application", host="0.0.0.0", port=8000, log_level="info")
    server = Server(config=config)
    await server.serve()

async def run_websocket_server():
    """
    Configures and starts the WebSocket server on port 5000.
    """
    config = Config("server.asgi_websocket:application", host="0.0.0.0", port=5000, log_level="info")
    server = Server(config=config)
    await server.serve()

async def listen_for_commands(shutdown_event: asyncio.Event):
    """
    Listens for user input to trigger server shutdown.
    """
    print("Type 'Quit' to stop the servers.")
    loop = asyncio.get_event_loop()
    while not shutdown_event.is_set():
        # Run the blocking input() function in an executor so it doesn't block the event loop.
        user_input = await loop.run_in_executor(None, input, "> ")
        if user_input.strip().lower() == "quit":
            print("Shutdown command received. Stopping servers...")
            shutdown_event.set()

async def main():
    shutdown_event = asyncio.Event()

    # Schedule both server tasks concurrently.
    http_task = asyncio.create_task(run_http_server())
    websocket_task = asyncio.create_task(run_websocket_server())
    command_task = asyncio.create_task(listen_for_commands(shutdown_event))

    # Wait until a shutdown command is received.
    await shutdown_event.wait()

    # Cancel the server and command tasks.
    http_task.cancel()
    websocket_task.cancel()
    command_task.cancel()

    # Optionally wait for tasks to finish their cleanup.
    try:
        await asyncio.gather(http_task, websocket_task, command_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    print("Servers have been shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
