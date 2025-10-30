from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import numpy as np

from src.mlvectordb.interfaces.query_processor import QueryProcessorProtocol
from src.mlvectordb.implementations.vector import Vector


class VectorRequest(BaseModel):
    id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None


class VectorBatchRequest(BaseModel):
    vectors: List[VectorRequest]


class SearchRequest(BaseModel):
    query: List[float]
    top_k: int
    namespace: str = "default"
    metric: str = "cosine"


class DeleteRequest(BaseModel):
    ids: List[str]
    namespace: str = "default"


class SearchResultResponse(BaseModel):
    id: str
    score: float
    vector: VectorRequest


class RestAPI:
    def __init__(self, query_processor: QueryProcessorProtocol):
        self.app = FastAPI(title="MLVectorDB REST API")
        self.query_processor = query_processor
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/insert")
        async def insert_vector(vector: VectorRequest, namespace: str = "default"):
            try:
                vector_obj = Vector(
                    id=vector.id,
                    values=vector.values,
                    metadata=vector.metadata or {}
                )
                self.query_processor.insert(vector_obj, namespace)
                return {"status": "success", "message": "Vector inserted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/upsert-many")
        async def upsert_vectors(batch: VectorBatchRequest, namespace: str = "default"):
            try:
                vectors = [
                    Vector(
                        id=vec.id,
                        values=vec.values,
                        metadata=vec.metadata or {}
                    )
                    for vec in batch.vectors
                ]
                self.query_processor.upsert_many(vectors, namespace)
                return {"status": "success", "message": f"{len(vectors)} vectors upserted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/search")
        async def search_similar(request: SearchRequest):
            try:
                query_array = np.array(request.query, dtype=np.float32)
                results = self.query_processor.find_similar(
                    query=query_array,
                    top_k=request.top_k,
                    namespace=request.namespace,
                    metric=request.metric
                )

                response_results = []
                for result in results:
                    response_results.append(SearchResultResponse(
                        id=result.id,
                        score=result.score,
                        vector=VectorRequest(
                            id=result.vector.id,
                            values=result.vector.values,
                            metadata=result.vector.metadata
                        )
                    ))

                return {
                    "status": "success",
                    "results": response_results,
                    "count": len(response_results)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/delete")
        async def delete_vectors(request: DeleteRequest):
            try:
                self.query_processor.delete(request.ids, request.namespace)
                return {
                    "status": "success",
                    "message": f"{len(request.ids)} vectors deleted from namespace '{request.namespace}'"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "MLVectorDB REST API"}

    def get_app(self):
        return self.app


def create_app(query_processor: QueryProcessorProtocol) -> FastAPI:
    api = RestAPI(query_processor)
    return api.get_app()
