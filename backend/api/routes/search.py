from ..models import QueryResponse, Document
from ..exceptions import SearchError
from ...core import config
from fastapi import APIRouter, Query, Request, HTTPException, status
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/search",
    response_model=QueryResponse,
    tags=["search"],
    summary="Search documents",
    description="Query search through indexed documents using different modes: boolean term, boolean wildcard(*), or TF-IDF search.",
    responses={
        200: {"description": "Search results returned successfully"},
        400: {"description": "Invalid search parameters"},
        500: {"description": "Search service unavailable"},
    },
)
def search(
    request: Request,
    query: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string",
        example="slee* cat",
    ),
    mode: str = Query(
        ...,
        pattern="^(term|wildcard|tfidf)$",
        description="Search mode: 'term', 'wildcard', or 'tfidf'",
        example="wildcard",
    ),
):
    """
    Perform query search with multiple search modes:

    - **term**: Boolean search with terms
    - **wildcard**: Boolean search with wildcard(*) pattern matching
    - **tfidf**: TF-IDF similarity-based search
    """
    inv = request.app.state.inv
    ret = request.app.state.ret

    # Check if search components are available
    if inv is None or ret is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service is currently unavailable. Please try again later.",
        )

    try:
        # Validate and clean query
        clean_query = query.strip()
        if not clean_query:
            raise SearchError("Query cannot be empty or contain only whitespace", query)

        logger.info(f"Processing search query: '{clean_query}' with mode: {mode}")

        # Perform search based on mode
        if mode == "term":
            terms = clean_query.split()
            if not terms:
                raise SearchError(
                    "Boolean term search requires at least one term", query
                )
            doc_ids = ret.terms_query(terms)
        elif mode == "wildcard":
            if len(clean_query) < 2:
                raise SearchError(
                    "Boolean wildcard(*) search requires at least 2 characters", query
                )
            doc_ids = ret.wildcard_and_query(
                clean_query, mode="bigram" if config.USE_BIGRAM else "permuterm"
            )
        elif mode == "tfidf":
            doc_ids = [
                doc_id
                for doc_id, _ in ret.query_similarity_top_k(clean_query, k=config.TOP_K)
            ]
        else:
            raise SearchError(f"Invalid search mode: {mode}", query)

        # Process results
        results = []
        for doc_id in doc_ids:
            try:
                row = inv.df[inv.df["docID"] == doc_id]
                if not row.empty:
                    tweet_id = str(row.iloc[0]["tweetID"])
                    text = row.iloc[0]["text"]
                    results.append(Document(tweet_id=tweet_id, text=text))
            except Exception as e:
                logger.warning(f"Error processing document {doc_id}: {str(e)}")
                continue

        logger.info(f"Search completed: {len(results)} results found")
        return QueryResponse(
            results=results, total_count=len(results), query=clean_query, mode=mode
        )

    except SearchError:
        raise  # Re-raise custom search errors
    except Exception as e:
        raise SearchError(f"Search operation failed: {str(e)}", query)
