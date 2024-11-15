import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lion.api.agent_api import app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add middleware to ensure proper error handling
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.exception("Error handling request")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Internal server error"},
        )


if __name__ == "__main__":
    try:
        # Configure host and port
        host = "127.0.0.1"
        port = 8000

        # Log startup information
        logger.info(
            f"API Key from environment: {os.environ.get('LION_API_KEY', 'Not set')}"
        )
        logger.info(f"Environment: {os.environ.get('LION_ENV', 'Not set')}")
        logger.info(f"Database URL: {os.environ.get('DATABASE_URL', 'Not set')}")
        logger.info(f"Server running at: http://{host}:{port}")
        logger.info("Press Ctrl+C to quit")

        # Run the server with reload enabled for development
        uvicorn.run(
            "lion.api.agent_api:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug",
            access_log=True,
        )
    except Exception as e:
        logger.exception("Failed to start server")
        raise
