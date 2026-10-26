"""
FastAPI application entrypoint for Crop Disease Detection backend.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
from .api.predict import router as predict_router
from .api.knowledge import router as knowledge_router
from .config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    KNOWLEDGE_BASE_PATH,
    MODEL_PATH,
)
from .services.inference import model_service
from .services.knowledge import knowledge_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the ML model and knowledge base
    once on startup and unloads them upon application shutdown.
    """
    logger.info("Initializing Crop Disease Detection backend...")
    logger.info(f"Target model path: {MODEL_PATH}")
    logger.info(f"Target knowledge base path: {KNOWLEDGE_BASE_PATH}")

    # Initialize ML model
    try:
        model_service.load_model(MODEL_PATH)
        logger.info("ML model initialized and ready for inference.")
    except Exception as exc:
        logger.warning(
            f"Model could not be loaded during startup: {exc}. "
            "Process will continue running; /health is operational and /ready will report 503."
        )

    # Initialize Agronomic Knowledge Base
    try:
        knowledge_service.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        logger.info("Knowledge base initialized and ready.")
    except Exception as exc:
        logger.warning(
            f"Knowledge base could not be loaded during startup: {exc}. "
            "Process will continue running; /health is operational and /api/knowledge will report 503."
        )

    yield

    logger.info("Shutting down backend, releasing resources...")
    model_service.unload_model()
    knowledge_service.unload()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    # Configure CORS with explicit allowed origins (never wildcard with credentials)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(knowledge_router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/health",
            "ready_url": "/ready",
            "predict_url": "/api/predict",
            "knowledge_url": "/api/knowledge/{class_id}",
        }

    return app


app = create_app()
