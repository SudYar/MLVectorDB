"""
Интерфейс ShardingManager для MLVectorDB.

Модуль определяет Protocol-интерфейс для управления шардированием данных
по нескольким шардам хранилища.
"""

from typing import Protocol, List, Optional, Dict, Any
from uuid import UUID

from typing_extensions import runtime_checkable

from .storage_engine import StorageEngine
from .vector import VectorProtocol


@runtime_checkable
class ShardingManager(Protocol):
    """Protocol-интерфейс для менеджеров шардирования в MLVectorDB."""
    
    @property
    def shard_count(self) -> int:
        """Количество активных шардов."""
        raise NotImplementedError
    
    def add_shard(self, shard_id: str, shard_url: Optional[str] = None) -> bool:
        """
        Добавить новый шард.
        
        Args:
            shard_id: Уникальный идентификатор шарда
            shard_url: URL шарда (опционально, для удаленных шардов)
            
        Returns:
            True если шард успешно добавлен
        """
        raise NotImplementedError
    
    def remove_shard(self, shard_id: str) -> bool:
        """
        Удалить шард.
        
        Args:
            shard_id: Идентификатор шарда для удаления
            
        Returns:
            True если шард успешно удален
        """
        raise NotImplementedError
    
    def get_shard_for_vector(
        self,
        vector: VectorProtocol,
        namespace: str = "default"
    ) -> str:
        """
        Определить шард для вектора.
        
        Args:
            vector: Вектор для размещения
            namespace: Пространство имен
            
        Returns:
            Идентификатор шарда
        """
        raise NotImplementedError
    
    def get_shard_for_id(
        self,
        vector_id: UUID,
        namespace: str = "default"
    ) -> Optional[str]:
        """
        Определить шард для вектора по его ID.
        
        Args:
            vector_id: ID вектора
            namespace: Пространство имен
            
        Returns:
            Идентификатор шарда или None если не найден
        """
        raise NotImplementedError
    
    def get_shards_for_search(
        self,
        namespace: str = "default"
    ) -> List[str]:
        """
        Получить список шардов для поиска (обычно все шарды).
        
        Args:
            namespace: Пространство имен
            
        Returns:
            Список идентификаторов шардов для поиска
        """
        raise NotImplementedError
    
    def redistribute_data(self, from_shard: str, to_shard: str) -> bool:
        """
        Перераспределить данные между шардами.
        
        Args:
            from_shard: Исходный шард
            to_shard: Целевой шард
            
        Returns:
            True если перераспределение успешно
        """
        raise NotImplementedError

    def redistribute_all_data(self) -> bool:
        raise NotImplementedError
    
    def check_shard_health(self, shard_id: str) -> bool:
        """
        Проверить работоспособность шарда.
        
        Args:
            shard_id: Идентификатор шарда
            
        Returns:
            True если шард жив и доступен
        """
        raise NotImplementedError
    
    def check_all_shards_health(self) -> Dict[str, bool]:
        """
        Проверить работоспособность всех шардов.
        
        Returns:
            Словарь {shard_id: is_healthy} со статусом всех шардов
        """
        raise NotImplementedError
    
    def get_shard_info(self) -> Dict[str, Any]:
        """
        Получить информацию о всех шардах.
        
        Returns:
            Словарь с информацией о шардах
        """
        raise NotImplementedError
    
    def list_shards(self) -> List[str]:
        """
        Получить список всех шардов.
        
        Returns:
            Список идентификаторов шардов
        """
        raise NotImplementedError

    def get_shard_storage(self, shard_id: str) -> Optional[StorageEngine]:
        raise NotImplementedError

    def update_shard_vector_count(self, shard_id: str) -> None:
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
