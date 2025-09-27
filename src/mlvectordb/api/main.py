"""
REST API for MLVectorDB using FastAPI.

This module provides a REST API interface to access the QueryProcessor
and other MLVectorDB functionality.
"""

from typing import Union, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from datetime import datetime

from ..implementations.basic_query_processor import BasicQueryProcessor
from .models import (
    QueryRequest, KNNQueryRequest, RangeQueryRequest, SimilarityQueryRequest,
    MetadataQueryRequest, HybridQueryRequest, QueryResponse, ErrorResponse,
    QueryExplanation, QueryStatistics, HealthCheckResponse, VectorData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLVectorDB API",
    description="REST API for MLVectorDB - A Vector Database implementation for Big Data Infrastructure course",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global query processor instance
query_processor = BasicQueryProcessor()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "MLVectorDB API",
        "version": "0.1.0",
        "description": "REST API for MLVectorDB Vector Database",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
        query_processor_status="active"
    )


@app.post("/query", response_model=QueryResponse)
async def execute_query(
    request: Union[
        KNNQueryRequest,
        RangeQueryRequest, 
        SimilarityQueryRequest,
        MetadataQueryRequest,
        HybridQueryRequest
    ] = Body(..., discriminator="type")
):
    """
    Execute a query against the vector database.
    
    Supports multiple query types:
    - knn: K-nearest neighbors search
    - range: Range search within radius
    - similarity: Similarity search with threshold
    - metadata: Metadata-based filtering
    - hybrid: Combined vector and metadata search
    """
    try:
        start_time = time.time()
        
        # Convert Pydantic model to dict
        query_dict = request.model_dump()
        
        # Execute query through the query processor
        result = query_processor.execute_query(query_dict)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        result["execution_time_ms"] = execution_time
        
        logger.info(f"Executed {request.type} query in {execution_time:.2f}ms")
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query/knn", response_model=QueryResponse)
async def knn_query(request: KNNQueryRequest):
    """Execute a K-nearest neighbors query."""
    return await execute_query(request)


@app.post("/query/range", response_model=QueryResponse) 
async def range_query(request: RangeQueryRequest):
    """Execute a range search query."""
    return await execute_query(request)


@app.post("/query/similarity", response_model=QueryResponse)
async def similarity_query(request: SimilarityQueryRequest):
    """Execute a similarity search query."""
    return await execute_query(request)


@app.post("/query/metadata", response_model=QueryResponse)
async def metadata_query(request: MetadataQueryRequest):
    """Execute a metadata-based query."""
    return await execute_query(request)


@app.post("/query/hybrid", response_model=QueryResponse)
async def hybrid_query(request: HybridQueryRequest):
    """Execute a hybrid vector and metadata query."""
    return await execute_query(request)


@app.post("/query/explain", response_model=QueryExplanation)
async def explain_query(
    request: Union[
        KNNQueryRequest,
        RangeQueryRequest,
        SimilarityQueryRequest, 
        MetadataQueryRequest,
        HybridQueryRequest
    ] = Body(..., discriminator="type")
):
    """
    Explain the execution plan for a query without executing it.
    """
    try:
        query_dict = request.model_dump()
        explanation = query_processor.explain_query(query_dict)
        
        return QueryExplanation(**explanation)
        
    except Exception as e:
        logger.error(f"Query explanation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/statistics", response_model=QueryStatistics)
async def get_statistics():
    """Get query processor statistics."""
    try:
        stats = query_processor.get_query_statistics()
        return QueryStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query result cache."""
    try:
        success = query_processor.clear_query_cache()
        return {"success": success, "message": "Query cache cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-types")
async def get_supported_query_types():
    """Get list of supported query types."""
    try:
        query_types = [qt.value for qt in query_processor.supported_query_types]
        return {
            "supported_query_types": query_types,
            "descriptions": {
                "knn": "K-nearest neighbors search",
                "range": "Range search within specified radius",
                "similarity": "Similarity search with threshold",
                "metadata": "Metadata-based filtering",
                "hybrid": "Combined vector and metadata search"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get query types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Example endpoint for testing
@app.post("/example/knn")
async def example_knn_query():
    """
    Example KNN query for testing purposes.
    Returns results for a sample 3-dimensional vector.
    """
    example_request = KNNQueryRequest(
        type="knn",
        vector=[1.0, 2.0, 3.0],
        k=5
    )
    
    return await execute_query(example_request)


@app.post("/example/range") 
async def example_range_query():
    """
    Example range query for testing purposes.
    """
    example_request = RangeQueryRequest(
        type="range",
        vector=[0.5, 1.5, 2.5],
        radius=2.0
    )
    
    return await execute_query(example_request)


@app.post("/example/similarity")
async def example_similarity_query():
    """
    Example similarity query for testing purposes.
    """
    example_request = SimilarityQueryRequest(
        type="similarity", 
        vector=[1.0, 0.0, 0.0],
        threshold=0.7,
        metric="cosine"
    )
    
    return await execute_query(example_request)


@app.post("/example/metadata")
async def example_metadata_query():
    """
    Example metadata query for testing purposes.
    """
    example_request = MetadataQueryRequest(
        type="metadata",
        filter={"category": "test", "active": True},
        limit=10
    )
    
    return await execute_query(example_request)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "error_type": "ValueError"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "error_type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mlvectordb.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )