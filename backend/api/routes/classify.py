from ...core.utils import preprocess
from ..exceptions import ClassificationError
from ..models import ClassificationResponse, ClassificationRequest
from fastapi import APIRouter, HTTPException, Request, status
import logging


router = APIRouter(tags=["classification"])
logger = logging.getLogger(__name__)


# === Claasify API ===
@router.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["classification"],
    summary="Classify text document",
    description="Classify a document using trained Naive Bayes classifier based on name, title, and review content in German.",
    responses={
        200: {"description": "Classification completed successfully"},
        400: {"description": "Invalid input data"},
        500: {"description": "Classification service unavailable"},
        503: {"description": "Classifier not trained or unavailable"},
    },
)
def classify(request: Request, req: ClassificationRequest):
    """
    Classify a document using Naive Bayes classifier.

    The classifier combines name, title, and review text to predict the document category("gut"(good)/"schlecht"(bad)).
    """
    cls = request.app.state.cls

    # Check if classifier is available
    if cls is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service is currently unavailable. Classifier not trained.",
        )

    try:
        # Combine and preprocess text
        combined_text = f"{req.name} {req.title} {req.review}".strip()
        if not combined_text:
            raise ClassificationError("Combined text cannot be empty")

        logger.info(
            f"Processing classification request for text length: {len(combined_text)}"
        )

        tokens = preprocess(combined_text)
        if not tokens:
            raise ClassificationError("No valid tokens found after preprocessing")

        label = cls.predict(tokens)
        if not label:
            raise ClassificationError("Classifier returned empty result")

        logger.info(f"Classification completed: {label}")
        return ClassificationResponse(label=label)

    except ClassificationError:
        raise  # Re-raise custom classification errors
    except Exception as e:
        raise ClassificationError(f"Classification failed: {str(e)}")
