from ...core import config
from ..exceptions import ClassificationError
from fastapi import APIRouter, HTTPException, Request, status
import logging
from datetime import datetime

router = APIRouter(tags=["classification"])
logger = logging.getLogger(__name__)


@router.get(
    "/classify/evaluate",
    tags=["classification"],
    summary="Evaluate classification model",
    description="Run evaluation on the classification model using test dataset and return accuracy and F1 score metrics.",
    responses={
        200: {"description": "Evaluation completed successfully"},
        500: {"description": "Evaluation process failed"},
        503: {"description": "Classifier not available for evaluation"},
    },
)
def evaluate(request: Request):
    """
    Evaluate the trained Naive Bayes classifier performance.

    Returns accuracy and F1 score metrics based on the test dataset.
    """
    cls = request.app.state.cls

    if cls is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification service is currently unavailable. Cannot perform evaluation.",
        )

    try:
        logger.info("Starting classifier evaluation...")

        cls.test(config.TEST_PATH)
        acc, f1 = cls.evaluate()

        if acc is None or f1 is None:
            raise ClassificationError("Evaluation returned invalid metrics")

        logger.info(f"Evaluation completed - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        return {
            "Accuracy": f"{acc:.4f}",
            "F1": f"{f1:.4f}",
            "timestamp": datetime.now().isoformat(),
        }

    except ClassificationError:
        raise  # Re-raise custom classification errors
    except Exception as e:
        raise ClassificationError(f"Evaluation failed: {str(e)}")
