"""
Пример использования репликации и шардирования в MLVectorDB.

Этот пример демонстрирует:
1. Настройку репликации
2. Настройку шардирования
3. Использование QueryProcessor с репликацией и шардированием
4. Управление репликами и шардами через API
"""

import sys
import time
from uuid import UUID

# Добавляем путь к проекту
sys.path.insert(0, '..')

from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory
from src.mlvectordb.implementations.index import Index
from src.mlvectordb.implementations.replication_manager import ReplicationManagerImpl
from src.mlvectordb.implementations.sharding_manager import ShardingManagerImpl
from src.mlvectordb.implementations.query_processor_with_replication import QueryProcessorWithReplication
from src.mlvectordb.interfaces.vector import VectorDTO


def example_replication():
    """Пример использования репликации."""
    print("=" * 60)
    print("Пример 1: Репликация")
    print("=" * 60)
    
    # Создаем первичное хранилище
    primary_storage = StorageEngineInMemory()
    index = Index()
    
    # Создаем менеджер репликации
    replication_manager = ReplicationManagerImpl(
        primary_storage=primary_storage,
        primary_replica_id="primary",
        health_check_interval=5.0
    )
    
    # Создаем QueryProcessor с репликацией
    qproc = QueryProcessorWithReplication(
        storage_engine=primary_storage,
        index=index,
        replication_manager=replication_manager
    )
    
    # Добавляем реплику (в реальном сценарии это будет URL другого сервера)
    print("\n1. Добавление реплики...")
    # В примере используем фиктивный URL, так как у нас нет реальных реплик
    # replication_manager.add_replica("replica_1", "http://localhost:8001")
    print("   Репликация настроена (для реального использования нужны запущенные реплики)")
    
    # Вставляем вектор
    print("\n2. Вставка вектора с репликацией...")
    vector = VectorDTO(
        values=[1.0, 2.0, 3.0, 4.0],
        metadata={"category": "test", "id": 1}
    )
    qproc.insert(vector, namespace="test")
    print(f"   Вектор вставлен: {vector.values}")
    
    # Проверяем информацию о репликации
    print("\n3. Информация о репликации:")
    info = replication_manager.get_replica_info()
    print(f"   Первичная реплика: {info['primary_replica']}")
    print(f"   Всего реплик: {info['total_replicas']}")
    print(f"   Здоровых реплик: {info['healthy_replicas']}")
    
    # Проверяем здоровье реплик
    print("\n4. Проверка здоровья реплик:")
    health = replication_manager.check_all_replicas_health()
    for replica_id, is_healthy in health.items():
        print(f"   {replica_id}: {'✓ Здорова' if is_healthy else '✗ Недоступна'}")


def example_sharding():
    """Пример использования шардирования."""
    print("\n" + "=" * 60)
    print("Пример 2: Шардирование")
    print("=" * 60)
    
    # Создаем первичное хранилище
    primary_storage = StorageEngineInMemory()
    index = Index()
    
    # Создаем локальные шарды
    shard_storages = {
        "shard_0": StorageEngineInMemory(),
        "shard_1": StorageEngineInMemory(),
        "shard_2": StorageEngineInMemory(),
    }
    
    # Создаем менеджер шардирования
    sharding_manager = ShardingManagerImpl(
        shard_storages=shard_storages,
        sharding_strategy="hash",
        health_check_interval=5.0
    )
    
    # Создаем QueryProcessor с шардированием
    qproc = QueryProcessorWithReplication(
        storage_engine=primary_storage,
        index=index,
        sharding_manager=sharding_manager
    )
    
    print("\n1. Вставка векторов с шардированием...")
    vectors = []
    for i in range(10):
        vector = VectorDTO(
            values=[float(i), float(i+1), float(i+2), float(i+3)],
            metadata={"id": i, "category": f"cat_{i % 3}"}
        )
        qproc.insert(vector, namespace="test")
        vectors.append(vector)
        # Получаем ID вставленного вектора из хранилища
        # В реальном использовании ID генерируется автоматически
        # Для демонстрации просто показываем распределение
        print(f"   Вектор {i} вставлен")
    
    # Информация о шардировании
    print("\n2. Информация о шардировании:")
    info = sharding_manager.get_shard_info()
    print(f"   Всего шардов: {info['total_shards']}")
    print(f"   Здоровых шардов: {info['healthy_shards']}")
    print(f"   Стратегия: {info['sharding_strategy']}")
    
    # Поиск
    print("\n3. Поиск похожих векторов:")
    query = VectorDTO(
        values=[5.0, 6.0, 7.0, 8.0],
        metadata={}
    )
    results = qproc.find_similar(query, top_k=5, namespace="test")
    print(f"   Найдено результатов: {len(results)}")
    for i, result in enumerate(results[:3]):
        print(f"   {i+1}. Score: {result['score']:.4f}, ID: {result['id']}")


def example_combined():
    """Пример комбинированного использования репликации и шардирования."""
    print("\n" + "=" * 60)
    print("Пример 3: Репликация + Шардирование")
    print("=" * 60)
    
    # Создаем первичное хранилище
    primary_storage = StorageEngineInMemory()
    index = Index()
    
    # Настраиваем репликацию
    replication_manager = ReplicationManagerImpl(
        primary_storage=primary_storage,
        primary_replica_id="primary",
        health_check_interval=5.0
    )
    
    # Настраиваем шардирование
    shard_storages = {
        "shard_0": StorageEngineInMemory(),
        "shard_1": StorageEngineInMemory(),
    }
    sharding_manager = ShardingManagerImpl(
        shard_storages=shard_storages,
        sharding_strategy="hash"
    )
    
    # Создаем QueryProcessor с обоими механизмами
    qproc = QueryProcessorWithReplication(
        storage_engine=primary_storage,
        index=index,
        replication_manager=replication_manager,
        sharding_manager=sharding_manager
    )
    
    print("\n1. Вставка векторов...")
    for i in range(5):
        vector = VectorDTO(
            values=[float(i) * 0.1, float(i+1) * 0.1, float(i+2) * 0.1],
            metadata={"id": i}
        )
        qproc.insert(vector, namespace="combined")
        print(f"   Вектор {i} вставлен")
    
    # Получаем полную информацию
    print("\n2. Полная информация о системе:")
    info = qproc.get_storage_info()
    print(f"   Хранилище: {info.get('storage_type', 'N/A')}")
    print(f"   Всего векторов: {info.get('total_vectors', 0)}")
    
    if "replication" in info:
        rep_info = info["replication"]
        print(f"   Репликация: {rep_info.get('healthy_replicas', 0)}/{rep_info.get('total_replicas', 0)} реплик")
    
    if "sharding" in info:
        shard_info = info["sharding"]
        print(f"   Шардирование: {shard_info.get('healthy_shards', 0)}/{shard_info.get('total_shards', 0)} шардов")


def main():
    """Главная функция с примерами."""
    print("\n" + "=" * 60)
    print("Примеры использования репликации и шардирования")
    print("=" * 60)
    
    try:
        # Пример 1: Только репликация
        example_replication()
        
        # Пример 2: Только шардирование
        example_sharding()
        
        # Пример 3: Комбинированное использование
        example_combined()
        
        print("\n" + "=" * 60)
        print("Все примеры выполнены успешно!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

