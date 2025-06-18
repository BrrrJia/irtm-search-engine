from pydantic import BaseModel
from typing import Optional, List
from fastapi import Request, status, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


# Custom Exception Classes
class SearchError(Exception):
    def __init__(self, message: str, query: str = None):
        self.message = message
        self.query = query
        super().__init__(self.message)


class ClassificationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ClusteringError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# Error Response Models
class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    path: Optional[str] = None
    query: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    validation_errors: List[dict]


# Global Exception Handlers
def register_exception_handlers(app: FastAPI):
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        logger.error(f"Validation error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ValidationErrorResponse(
                error="Validation Error",
                detail="Invalid input data provided",
                timestamp=datetime.now().isoformat(),
                validation_errors=[
                    {
                        "field": (
                            err.get("loc", [])[-1] if err.get("loc") else "unknown"
                        ),
                        "message": err.get("msg", ""),
                        "type": err.get("type", ""),
                    }
                    for err in exc.errors()
                ],
            ).dict(),
        )

    @app.exception_handler(SearchError)
    async def search_exception_handler(request: Request, exc: SearchError):
        logger.error(f"Search error: {exc.message} - Query: {exc.query}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="Search Error",
                detail=exc.message,
                timestamp=datetime.now().isoformat(),
                path=str(request.url.path),
                query=exc.query,
            ).dict(),
        )

    @app.exception_handler(ClassificationError)
    async def classification_exception_handler(
        request: Request, exc: ClassificationError
    ):
        logger.error(f"Classification error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Classification Error",
                detail=exc.message,
                timestamp=datetime.now().isoformat(),
                path=str(request.url.path),
            ).dict(),
        )

    @app.exception_handler(ClusteringError)
    async def clustering_exception_handler(request: Request, exc: ClusteringError):
        logger.error(f"Clustering error: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Clustering Error",
                detail=exc.message,
                timestamp=datetime.now().isoformat(),
                path=str(request.url.path),
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error on {request.url.path}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred. Please try again later.",
                timestamp=datetime.now().isoformat(),
                path=str(request.url.path),
            ).dict(),
        )
