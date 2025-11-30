"""
Интерфейс ReplicationManager для MLVectorDB.

Модуль определяет Protocol-интерфейс для управления репликацией данных
между несколькими репликами хранилища.
"""

from typing import Protocol, List, Optional, Dict, Any, Sequence
from uuid import UUID
from typing_extensions import runtime_checkable

from .vector import VectorProtocol, VectorDTO


@runtime_checkable
class ReplicationManager(Protocol):
    """Protocol-интерфейс для менеджеров репликации в MLVectorDB."""
    
    @property
    def replica_count(self) -> int:
        """Количество активных реплик."""
        raise NotImplementedError
    
    @property
    def primary_replica(self) -> Optional[str]:
        """Идентификатор первичной реплики."""
        raise NotImplementedError
    
    def add_replica(self, replica_id: str, replica_url: str) -> bool:
        """
        Добавить новую реплику.
        
        Args:
            replica_id: Уникальный идентификатор реплики
            replica_url: URL реплики для синхронизации
            
        Returns:
            True если реплика успешно добавлена
        """
        raise NotImplementedError
    
    def remove_replica(self, replica_id: str) -> bool:
        """
        Удалить реплику.
        
        Args:
            replica_id: Идентификатор реплики для удаления
            
        Returns:
            True если реплика успешно удалена
        """
        raise NotImplementedError
    
    def replicate_write(
        self,
        vector: VectorProtocol,
        namespace: str
    ) -> Dict[str, bool]:
        """
        Реплицировать запись вектора на все реплики.
        
        Args:
            vector: Вектор для репликации
            namespace: Пространство имен
            
        Returns:
            Словарь {replica_id: success} с результатами репликации
        """
        raise NotImplementedError
    
    def replicate_delete(
        self,
        vector_id: UUID,
        namespace: str
    ) -> Dict[str, bool]:
        """
        Реплицировать удаление вектора на все реплики.
        
        Args:
            vector_id: ID вектора для удаления
            namespace: Пространство имен
            
        Returns:
            Словарь {replica_id: success} с результатами репликации
        """
        raise NotImplementedError
    
    def replicate_batch_write(
        self,
        vectors: Sequence[VectorProtocol],
        namespace: str
    ) -> Dict[str, bool]:
        """
        Реплицировать пакетную запись векторов на все реплики.
        
        Args:
            vectors: Список векторов для репликации
            namespace: Пространство имен
            
        Returns:
            Словарь {replica_id: success} с результатами репликации
        """
        raise NotImplementedError
    
    def check_replica_health(self, replica_id: str) -> bool:
        """
        Проверить работоспособность реплики.
        
        Args:
            replica_id: Идентификатор реплики
            
        Returns:
            True если реплика жива и доступна
        """
        raise NotImplementedError
    
    def check_all_replicas_health(self) -> Dict[str, bool]:
        """
        Проверить работоспособность всех реплик.
        
        Returns:
            Словарь {replica_id: is_healthy} со статусом всех реплик
        """
        raise NotImplementedError
    
    def sync_replica(self, replica_id: str, namespace: Optional[str] = None) -> bool:
        """
        Синхронизировать реплику с первичной.
        
        Args:
            replica_id: Идентификатор реплики для синхронизации
            namespace: Пространство имен (None для всех)
            
        Returns:
            True если синхронизация успешна
        """
        raise NotImplementedError
    
    def get_replica_info(self) -> Dict[str, Any]:
        """
        Получить информацию о всех репликах.
        
        Returns:
            Словарь с информацией о репликах
        """
        raise NotImplementedError
    
    def list_replicas(self) -> List[str]:
        """
        Получить список всех реплик.
        
        Returns:
            Список идентификаторов реплик
        """
        raise NotImplementedError

