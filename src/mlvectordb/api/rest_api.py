import logging
import sys
import time
from uuid import UUID
from typing import List, Any, Dict
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
                self.query_processor.delete(delete_request.ids, namespace)

                self.logger.info(
                    f"Удаление завершено успешно - "
                    f"{len(delete_request.ids)} векторов удалено из пространства: {namespace}"
                )
                return {
                    "status": "success",
                    "message": f"{len(delete_request.ids)} vectors deleted"
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
