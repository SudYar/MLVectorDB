import logging
import sys
import time
from uuid import UUID
from typing import List, Any, Dict, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, status, Request, Query
from pydantic import BaseModel, Field

from src.mlvectordb.interfaces.query_processor import QueryProcessorProtocol
from src.mlvectordb.interfaces.vector import VectorDTO


# Pydantic модели для API (из второго файла)
class VectorCreateRequest(BaseModel):
    values: List[float] = Field(..., description="Вектор как список чисел")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные вектора")


class VectorSearchRequest(BaseModel):
    query: List[float] = Field(..., description="Вектор запроса")
    top_k: int = Field(10, ge=1, le=1000, description="Количество возвращаемых результатов")
    metric: str = Field("cosine", description="Метрика расстояния")


class VectorSearchResult(BaseModel):
    id: UUID
    values: List[float]
    metadata: Dict[str, Any]
    score: float


class VectorDeleteRequest(BaseModel):
    ids: List[UUID] = Field(..., description="Список ID векторов для удаления")


class BatchVectorRequest(BaseModel):
    vectors: List[VectorCreateRequest] = Field(..., description="Список векторов")


class VectorInfo(BaseModel):
    id: UUID
    values: List[float]
    metadata: Dict[str, Any]


class RestAPI:
    def __init__(
            self,
            query_processor: QueryProcessorProtocol,
            title: str = "Vector DB API",
            enable_file_logging: bool = False,
            log_level: str = "INFO"
    ):
        """
        Инициализация REST API

        Args:
            query_processor: Процессор запросов к векторной БД
            title: Заголовок API
            enable_file_logging: Включить файловое логирование
            log_level: Уровень логирования
        """
        self.query_processor = query_processor
        self.title = title
        self.enable_file_logging = enable_file_logging

        # Настройка логирования
        self._setup_logging(log_level)
        self.logger = logging.getLogger("vector_db_api")

        # Создание приложения с lifespan контекстом
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Запущен Vector DB API...")
            yield
            # Shutdown
            self.logger.info("Остановлен Vector DB API...")

        self.app = FastAPI(
            title=self.title,
            lifespan=lifespan
        )

        # Настройка middleware и маршрутов
        self._setup_middleware()
        self._setup_routes()


    def _setup_routes(self):
        """Настройка маршрутов API"""

        @self.app.post("/vectors", status_code=status.HTTP_201_CREATED)
        async def insert_vector(
                vector: VectorCreateRequest,
                namespace: str = Query("default", description="Namespace for the vector")
        ):
            """Вставка одного вектора"""
            self.logger.info(
                f"Запрос на вставку вектора - "
                f"пространство: {namespace}, размер: {len(vector.values)}, "
                f"метаданные: {list(vector.metadata.keys())}"
            )

            try:
                vector_dto = VectorDTO(values=vector.values, metadata=vector.metadata)
                self.logger.debug("Вызов query_processor.insert")
                self.query_processor.insert(vector_dto, namespace)

                self.logger.info(f"Вектор успешно вставлен в пространство: {namespace}")
                return {"status": "success", "message": "Vector inserted"}

            except Exception as e:
                self.logger.error(
                    f"Ошибка вставки - пространство: {namespace}, ошибка: {str(e)}",
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Insert failed: {str(e)}"
                )

        @self.app.put("/vectors/batch")
        async def upsert_vectors(
                batch_request: BatchVectorRequest,
                namespace: str = Query("default", description="Namespace for the vectors")
        ):
            """Массовый upsert векторов"""
            self.logger.info(
                f"Запрос массового upsert - "
                f"пространство: {namespace}, количество векторов: {len(batch_request.vectors)}"
            )

            try:
                vectors_dto = [
                    VectorDTO(values=vector.values, metadata=vector.metadata)
                    for vector in batch_request.vectors
                ]
                self.logger.debug(f"Вызов query_processor.upsert_many с {len(vectors_dto)} векторами")
                self.query_processor.upsert_many(vectors_dto, namespace)

                self.logger.info(
                    f"Массовый upsert завершен успешно - "
                    f"{len(vectors_dto)} векторов в пространстве: {namespace}"
                )
                return {"status": "success", "message": f"{len(vectors_dto)} vectors upserted"}

            except Exception as e:
                self.logger.error(
                    f"Ошибка массового upsert - "
                    f"пространство: {namespace}, количество векторов: {len(batch_request.vectors)}, "
                    f"ошибка: {str(e)}",
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch upsert failed: {str(e)}"
                )

        @self.app.post("/search", response_model=List[VectorSearchResult])
        async def search_similar(
                search_request: VectorSearchRequest,
                namespace: str = Query("default", description="Namespace to search in")
        ):
            """Поиск похожих векторов"""
            self.logger.info(
                f"Запрос поиска - "
                f"пространство: {namespace}, top_k: {search_request.top_k}, "
                f"метрика: {search_request.metric}, "
                f"размер вектора запроса: {len(search_request.query)}"
            )

            try:
                query_dto = VectorDTO(values=search_request.query, metadata={})
                self.logger.debug("Вызов query_processor.find_similar")
                results = self.query_processor.find_similar(
                    query=query_dto,
                    top_k=search_request.top_k,
                    namespace=namespace,
                    metric=search_request.metric
                )

                self.logger.info(
                    f"Поиск завершен - найдено {len(results)} результатов в пространстве: {namespace}"
                )
                if results:
                    scores = [r['score'] for r in results]
                    self.logger.debug(f"Результаты поиска (оценки): {scores}")

                return results

            except Exception as e:
                self.logger.error(
                    f"Ошибка поиска - пространство: {namespace}, ошибка: {str(e)}",
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Search failed: {str(e)}"
                )

        @self.app.delete("/vectors")
        async def delete_vectors(
                delete_request: VectorDeleteRequest,
                namespace: str = Query("default", description="Namespace to delete from")
        ):
            """Удаление векторов по ID"""
            self.logger.info(
                f"Запрос удаления векторов - "
                f"пространство: {namespace}, количество ID: {len(delete_request.ids)}"
            )

            if not delete_request.ids:
                self.logger.warning("Запрос удаления с пустым списком ID")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No IDs provided"
                )

            try:
                del_uuid = self.query_processor.delete(delete_request.ids, namespace)

                self.logger.info(
                    f"Удаление завершено успешно - "
                    f"{len(del_uuid)} векторов удалено из пространства: {namespace}"
                )
                self.logger.debug(f"uuid удаленных векторов: {del_uuid}")
                return {
                    "status": f"{'success' if len(del_uuid) else 'error'}",
                    "message": f"{len(del_uuid)} vectors deleted"
                }

            except Exception as e:
                self.logger.error(
                    f"Ошибка удаления - "
                    f"пространство: {namespace}, количество ID: {len(delete_request.ids)}, "
                    f"ошибка: {str(e)}",
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Delete failed: {str(e)}"
                )

        @self.app.get("/namespaces")
        async def list_namespaces():
            """Получение списка всех пространств имен"""
            self.logger.info("Запрос на получение списка пространств имен")
            try:
                namespaces = self.query_processor.list_namespaces()
                self.logger.info(f"Найдено {len(namespaces)} пространств: {namespaces}")
                return {"namespaces": namespaces}
            except Exception as e:
                self.logger.error(f"Ошибка получения списка пространств: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list namespaces: {str(e)}"
                )

        @self.app.get("/namespaces/vectors", response_model=List[VectorInfo])
        async def get_namespace_vectors(namespace: str = Query("default", description="Namespace vectors from")):
            """Получение всех векторов в пространстве имен"""
            self.logger.info(f"Запрос на получение векторов в пространстве: {namespace}")
            try:
                vectors = self.query_processor.get_namespace_vectors(namespace)
                self.logger.info(f"Найдено {len(vectors)} векторов в пространстве: {namespace}")
                return vectors
            except Exception as e:
                self.logger.error(f"Ошибка получения векторов для пространства {namespace}: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get vectors: {str(e)}"
                )

        @self.app.get("/storage/info")
        async def get_storage_info():
            """Получение информации о хранилище"""
            self.logger.info("Запрос на получение информации о хранилище")
            try:
                info = self.query_processor.get_storage_info()
                self.logger.info(f"Информация о хранилище: {info}")
                return info
            except Exception as e:
                self.logger.error(f"Ошибка получения информации о хранилище: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get storage info: {str(e)}"
                )
        @self.app.get("/health")
        async def health_check():
            """Проверка работоспособности API"""
            self.logger.debug("Проверка здоровья API")
            return {"status": "healthy"}

        @self.app.post("/log/level")
        async def set_log_level(level: str):
            """Изменение уровня логирования (debug, info, warning, error)"""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            if level.upper() not in valid_levels:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid level. Must be one of: {valid_levels}"
                )

            logging.getLogger().setLevel(level.upper())
            self.logger.info(f"Уровень логирования изменен на: {level.upper()}")

            return {"status": "success", "message": f"Log level set to {level.upper()}"}

        # Endpoints для управления репликацией
        @self.app.get("/replication/info")
        async def get_replication_info():
            """Получить информацию о репликации"""
            self.logger.info("Запрос информации о репликации")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    info = self.query_processor._replication_manager.get_replica_info()
                    return info
                else:
                    return {"message": "Репликация не настроена"}
            except Exception as e:
                self.logger.error(f"Ошибка получения информации о репликации: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get replication info: {str(e)}"
                )

        @self.app.post("/replication/replicas")
        async def add_replica(replica_id: str = Query(..., description="ID реплики"), 
                             replica_url: str = Query(..., description="URL реплики")):
            """Добавить новую реплику"""
            self.logger.info(f"Запрос на добавление реплики: {replica_id} по адресу {replica_url}")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    success = self.query_processor._replication_manager.add_replica(replica_id, replica_url)
                    if success:
                        return {"status": "success", "message": f"Реплика {replica_id} добавлена"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Не удалось добавить реплику {replica_id}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Репликация не настроена"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка добавления реплики: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to add replica: {str(e)}"
                )

        @self.app.delete("/replication/replicas/{replica_id}")
        async def remove_replica(replica_id: str):
            """Удалить реплику"""
            self.logger.info(f"Запрос на удаление реплики: {replica_id}")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    success = self.query_processor._replication_manager.remove_replica(replica_id)
                    if success:
                        return {"status": "success", "message": f"Реплика {replica_id} удалена"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Реплика {replica_id} не найдена"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Репликация не настроена"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка удаления реплики: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to remove replica: {str(e)}"
                )

        @self.app.get("/replication/replicas/{replica_id}/health")
        async def check_replica_health(replica_id: str):
            """Проверить здоровье реплики"""
            self.logger.info(f"Запрос проверки здоровья реплики: {replica_id}")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    is_healthy = self.query_processor._replication_manager.check_replica_health(replica_id)
                    return {"replica_id": replica_id, "is_healthy": is_healthy}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Репликация не настроена"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка проверки здоровья реплики: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to check replica health: {str(e)}"
                )

        @self.app.get("/replication/replicas/health")
        async def check_all_replicas_health():
            """Проверить здоровье всех реплик"""
            self.logger.info("Запрос проверки здоровья всех реплик")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    health_status = self.query_processor._replication_manager.check_all_replicas_health()
                    return {"replicas_health": health_status}
                else:
                    return {"message": "Репликация не настроена"}
            except Exception as e:
                self.logger.error(f"Ошибка проверки здоровья реплик: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to check replicas health: {str(e)}"
                )

        @self.app.post("/replication/replicas/{replica_id}/sync")
        async def sync_replica(replica_id: str, namespace: Optional[str] = Query(None, description="Namespace для синхронизации (опционально)")):
            """Синхронизировать реплику с первичной"""
            self.logger.info(f"Запрос синхронизации реплики: {replica_id}")
            try:
                if hasattr(self.query_processor, '_replication_manager') and self.query_processor._replication_manager:
                    success = self.query_processor._replication_manager.sync_replica(replica_id, namespace)
                    if success:
                        return {"status": "success", "message": f"Реплика {replica_id} синхронизирована"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Не удалось синхронизировать реплику {replica_id}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Репликация не настроена"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка синхронизации реплики: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to sync replica: {str(e)}"
                )

        # Endpoints для управления шардированием
        @self.app.get("/sharding/info")
        async def get_sharding_info():
            """Получить информацию о шардировании"""
            self.logger.info("Запрос информации о шардировании")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    info = self.query_processor._sharding_manager.get_shard_info()
                    return info
                else:
                    return {"message": "Шардирование не настроено"}
            except Exception as e:
                self.logger.error(f"Ошибка получения информации о шардировании: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get sharding info: {str(e)}"
                )

        @self.app.post("/sharding/shards")
        async def add_shard(shard_id: str = Query(..., description="ID шарда"),
                           shard_url: str = Query(None, description="URL шарда (опционально)")):
            """Добавить новый шард"""
            self.logger.info(f"Запрос на добавление шарда: {shard_id}")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    success = self.query_processor._sharding_manager.add_shard(shard_id, shard_url)
                    if success:
                        return {"status": "success", "message": f"Шард {shard_id} добавлен"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Не удалось добавить шард {shard_id}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Шардирование не настроено"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка добавления шарда: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to add shard: {str(e)}"
                )

        @self.app.post("/sharding/shards/local")
        async def add_local_shard(shard_id: str = Query(..., description="ID локального шарда")):
            """Добавить новый локальный шард с новым StorageEngine"""
            self.logger.info(f"Запрос на добавление локального шарда: {shard_id}")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    # Создаем новый StorageEngineInMemory для локального шарда
                    from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory
                    new_storage = StorageEngineInMemory()
                    
                    success = self.query_processor._sharding_manager.add_local_shard(shard_id, new_storage)
                    if success:
                        return {
                            "status": "success", 
                            "message": f"Локальный шард {shard_id} добавлен",
                            "shard_id": shard_id,
                            "is_local": True
                        }
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Не удалось добавить локальный шард {shard_id}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Шардирование не настроено"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка добавления локального шарда: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to add local shard: {str(e)}"
                )

        @self.app.delete("/sharding/shards/{shard_id}")
        async def remove_shard(shard_id: str):
            """Удалить шард"""
            self.logger.info(f"Запрос на удаление шарда: {shard_id}")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    success = self.query_processor._sharding_manager.remove_shard(shard_id)
                    if success:
                        return {"status": "success", "message": f"Шард {shard_id} удален"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Шард {shard_id} не найден или это последний шард"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Шардирование не настроено"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка удаления шарда: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to remove shard: {str(e)}"
                )

        @self.app.get("/sharding/shards/{shard_id}/health")
        async def check_shard_health(shard_id: str):
            """Проверить здоровье шарда"""
            self.logger.info(f"Запрос проверки здоровья шарда: {shard_id}")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    is_healthy = self.query_processor._sharding_manager.check_shard_health(shard_id)
                    return {"shard_id": shard_id, "is_healthy": is_healthy}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Шардирование не настроено"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка проверки здоровья шарда: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to check shard health: {str(e)}"
                )

        @self.app.get("/sharding/shards/health")
        async def check_all_shards_health():
            """Проверить здоровье всех шардов"""
            self.logger.info("Запрос проверки здоровья всех шардов")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    health_status = self.query_processor._sharding_manager.check_all_shards_health()
                    return {"shards_health": health_status}
                else:
                    return {"message": "Шардирование не настроено"}
            except Exception as e:
                self.logger.error(f"Ошибка проверки здоровья шардов: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to check shards health: {str(e)}"
                )

        @self.app.post("/sharding/shards/{from_shard}/redistribute/{to_shard}")
        async def redistribute_shard_data(from_shard: str, to_shard: str):
            """Перераспределить данные между шардами"""
            self.logger.info(f"Запрос перераспределения данных: {from_shard} -> {to_shard}")
            try:
                if hasattr(self.query_processor, '_sharding_manager') and self.query_processor._sharding_manager:
                    success = self.query_processor._sharding_manager.redistribute_data(from_shard, to_shard)
                    if success:
                        return {"status": "success", "message": f"Данные перераспределены из {from_shard} в {to_shard}"}
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Не удалось перераспределить данные из {from_shard} в {to_shard}"
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Шардирование не настроено"
                    )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка перераспределения данных: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to redistribute data: {str(e)}"
                )

    def get_app(self) -> FastAPI:
        """Получение FastAPI приложения"""
        return self.app

    def _setup_logging(self, log_level: str):
        """Настройка единого формата логирования"""

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                record.asctime_formatted = self.formatTime(record, self.datefmt)
                return super().format(record)

        LOG_FORMAT = '%(asctime_formatted)s - %(name)s - %(levelname)s - %(message)s'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Удаляем существующие обработчики
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Создаем и настраиваем обработчики
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = CustomFormatter(LOG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if self.enable_file_logging:
            file_handler = logging.FileHandler('vector_db_api.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def _setup_middleware(self):
        """Настройка middleware для логирования запросов"""

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()

            self.logger.info(f"→ Входящий запрос: {request.method} {request.url.path}")

            # Логируем тело запроса для отладки
            if request.method in ["POST", "PUT"] and self.logger.isEnabledFor(logging.DEBUG):
                try:
                    body = await request.body()
                    if len(body) < 1000:
                        self.logger.debug(f"Тело запроса: {body.decode()}")

                    async def receive():
                        return {'type': 'http.request', 'body': body}

                    request._receive = receive
                except Exception as e:
                    self.logger.warning(f"Не удалось прочитать тело запроса: {e}")

            response = await call_next(request)

            process_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"← Ответ: {request.method} {request.url.path} - "
                f"Статус: {response.status_code} - Время: {process_time:.2f}мс"
            )

            return response


# Пример использования
if __name__ == "__main__":
    # Пример инициализации
    from src.mlvectordb.implementations.query_processor import QueryProcessor
    from src.mlvectordb import Index, StorageEngineInMemory

    # Инициализация компонентов
    qproc = QueryProcessor(StorageEngineInMemory(), Index())

    # Создание API с кастомными настройками
    api = RestAPI(
        query_processor=qproc,
        title="MLVectorDB Production API",
        enable_file_logging=True,
        log_level="INFO"
    )

    # Запуск сервера
    uvicorn.run(
        api.get_app(),
        host="127.0.0.1",
        port=8000,
        log_config=None
    )
