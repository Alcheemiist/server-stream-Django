# runservers.py

import asyncio
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

async def run_http_server():
    """
    Configures and starts the HTTP server.
    """
    config = Config("server.asgi_http:application", host="0.0.0.0", port=8000, log_level="info")
    server = Server(config=config)
    await server.serve()

async def run_websocket_server():
    """
    Configures and starts the WebSocket server.
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
        # Run the blocking input function in a separate thread to prevent blocking the event loop
        user_input = await loop.run_in_executor(None, input, "> ")
        if user_input.strip().lower() == "quit":
            print("Shutdown command received. Stopping servers...")
            shutdown_event.set()

async def main():
    shutdown_event = asyncio.Event()

    # Schedule server coroutines
    http_task = asyncio.create_task(run_http_server())
    websocket_task = asyncio.create_task(run_websocket_server())
    command_task = asyncio.create_task(listen_for_commands(shutdown_event))

    # Wait until shutdown_event is set
    await shutdown_event.wait()

    # Initiate server shutdown
    http_task.cancel()
    websocket_task.cancel()
    command_task.cancel()

    # Optionally, wait for tasks to finish cleanup
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
