from ...core import config
from ...core.clustering import k_means, optimal_k_means, plot_clusters
from ..exceptions import ClusteringError
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import logging
import numpy as np

router = APIRouter(tags=["clustering"])
logger = logging.getLogger(__name__)


@router.get(
    "/clustering",
    tags=["clustering"],
    summary="Perform document clustering",
    description="Execute K-means clustering on document vectors and return visualization plot as PNG image.",
    responses={
        200: {
            "description": "Clustering visualization generated successfully",
            "content": {"image/png": {}},
        },
        500: {"description": "Clustering process failed"},
        503: {"description": "Clustering service unavailable"},
    },
)
def clustering(request: Request):
    """
    Perform K-means clustering on document data and return visualization.

    - Uses optimal K-means if configured, otherwise standard K-means
    - Highlights clusters with RSS values between 0 and 1
    - Returns clustering plot as PNG image stream
    """
    data = request.app.state.data

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clustering service is currently unavailable. Data not prepared.",
        )

    try:
        logger.info("Starting clustering operation...")

        if config.USE_OPTIMAL_KMEANS:
            classes, centroids, _, rss = optimal_k_means(
                data, k=config.K_MEANS_K, n_init=config.K_MEANS_N_INIT
            )
            title = "Optimal K-means Clustering"
        else:
            classes, centroids, _, rss = k_means(data, k=config.K_MEANS_K)
            title = "K-means Clustering"

        if classes is None or centroids is None:
            raise ClusteringError("Clustering algorithm returned invalid results")

        # Find clusters with small RSS values
        small_rss_cluster = np.where((rss > 0) & (rss < 1))[0]

        # Generate plot
        buf = plot_clusters(
            data,
            classes,
            centroids=centroids,
            title=title,
            highlight=small_rss_cluster,
            save_to_file=False,
        )

        if buf is None:
            raise ClusteringError("Failed to generate clustering plot")

        logger.info("Clustering visualization generated successfully")
        return StreamingResponse(buf, media_type="image/png")

    except ClusteringError:
        raise  # Re-raise custom clustering errors
    except Exception as e:
        raise ClusteringError(f"Clustering operation failed: {str(e)}")
