# Репликация и Шардирование в MLVectorDB

Этот документ описывает реализацию репликации и шардирования в MLVectorDB.

## Обзор

MLVectorDB поддерживает два механизма для масштабирования и отказоустойчивости:

1. **Репликация** - синхронизация данных между несколькими репликами для повышения отказоустойчивости
2. **Шардирование** - распределение данных по нескольким шардам для горизонтального масштабирования

## Репликация

### Описание

Репликация позволяет синхронизировать данные между несколькими экземплярами базы данных. При записи данных они автоматически реплицируются на все настроенные реплики.

### Особенности

- ✅ Динамическое добавление/удаление реплик без перезапуска
- ✅ Автоматическая проверка здоровья реплик
- ✅ Асинхронная репликация (не блокирует операции записи)
- ✅ Обработка ошибок репликации
- ✅ Поддержка батч-операций

### Использование

#### Программный API

```python
from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory
from src.mlvectordb.implementations.index import Index
from src.mlvectordb.implementations.replication_manager import ReplicationManagerImpl
from src.mlvectordb.implementations.query_processor_with_replication import QueryProcessorWithReplication

# Создание менеджера репликации
primary_storage = StorageEngineInMemory()
replication_manager = ReplicationManagerImpl(
    primary_storage=primary_storage,
    primary_replica_id="primary",
    health_check_interval=5.0  # Проверка здоровья каждые 5 секунд
)

# Добавление реплики
replication_manager.add_replica("replica_1", "http://localhost:8001")
replication_manager.add_replica("replica_2", "http://localhost:8002")

# Создание QueryProcessor с репликацией
qproc = QueryProcessorWithReplication(
    storage_engine=primary_storage,
    index=Index(),
    replication_manager=replication_manager
)

# Вставка вектора (автоматически реплицируется)
vector = VectorDTO(values=[1.0, 2.0, 3.0], metadata={})
qproc.insert(vector, namespace="test")
```

#### REST API

```bash
# Получить информацию о репликации
curl http://localhost:8000/replication/info

# Добавить реплику
curl -X POST "http://localhost:8000/replication/replicas?replica_id=replica_1&replica_url=http://localhost:8001"

# Удалить реплику
curl -X DELETE http://localhost:8000/replication/replicas/replica_1

# Проверить здоровье реплики
curl http://localhost:8000/replication/replicas/replica_1/health

# Проверить здоровье всех реплик
curl http://localhost:8000/replication/replicas/health
```

### Запуск сервера с репликацией

```bash
python -m src.mlvectordb.api.server --enable-replication --port 8000
```

## Шардирование

### Описание

Шардирование распределяет данные по нескольким шардам на основе хеширования ID вектора. Это позволяет горизонтально масштабировать систему.

### Особенности

- ✅ Динамическое добавление/удаление шардов
- ✅ Автоматическое определение шарда для вектора
- ✅ Поддержка локальных и удаленных шардов
- ✅ Автоматическая проверка здоровья шардов
- ✅ Агрегация результатов поиска из всех шардов

### Использование

#### Программный API

```python
from src.mlvectordb.implementations.sharding_manager import ShardingManagerImpl
from src.mlvectordb.implementations.storage_engine_in_memory import StorageEngineInMemory

# Создание локальных шардов
shard_storages = {
    "shard_0": StorageEngineInMemory(),
    "shard_1": StorageEngineInMemory(),
    "shard_2": StorageEngineInMemory(),
}

# Создание менеджера шардирования
sharding_manager = ShardingManagerImpl(
    shard_storages=shard_storages,
    sharding_strategy="hash",  # или "round_robin"
    health_check_interval=5.0
)

# Добавление удаленного шарда
sharding_manager.add_shard("shard_3", "http://localhost:8003")

# Создание QueryProcessor с шардированием
qproc = QueryProcessorWithReplication(
    storage_engine=primary_storage,
    index=Index(),
    sharding_manager=sharding_manager
)

# Вставка вектора (автоматически распределяется по шардам)
vector = VectorDTO(values=[1.0, 2.0, 3.0], metadata={})
qproc.insert(vector, namespace="test")

# Определение шарда для вектора
shard_id = sharding_manager.get_shard_for_vector(vector, namespace="test")
print(f"Вектор будет храниться в шарде: {shard_id}")
```

#### REST API

```bash
# Получить информацию о шардировании
curl http://localhost:8000/sharding/info

# Добавить шард
curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_3&shard_url=http://localhost:8003"

# Удалить шард
curl -X DELETE http://localhost:8000/sharding/shards/shard_3

# Проверить здоровье шарда
curl http://localhost:8000/sharding/shards/shard_3/health

# Проверить здоровье всех шардов
curl http://localhost:8000/sharding/shards/health
```

### Запуск сервера с шардированием

```bash
python -m src.mlvectordb.api.server --enable-sharding --port 8000
```

## Комбинированное использование

Репликацию и шардирование можно использовать вместе:

```python
# Настройка обоих механизмов
replication_manager = ReplicationManagerImpl(...)
sharding_manager = ShardingManagerImpl(...)

qproc = QueryProcessorWithReplication(
    storage_engine=primary_storage,
    index=Index(),
    replication_manager=replication_manager,
    sharding_manager=sharding_manager
)
```

При вставке вектора:
1. Вектор распределяется по соответствующему шарду (шардирование)
2. Данные реплицируются на все реплики (репликация)

При поиске:
1. Запрос выполняется на всех шардах
2. Результаты агрегируются и сортируются

## Health Check

Оба менеджера автоматически проверяют здоровье реплик/шардов в фоновом режиме.

### Настройка интервала проверки

```python
# Репликация
replication_manager = ReplicationManagerImpl(
    primary_storage=storage,
    health_check_interval=5.0  # Проверка каждые 5 секунд
)

# Шардирование
sharding_manager = ShardingManagerImpl(
    shard_storages=shards,
    health_check_interval=10.0  # Проверка каждые 10 секунд
)
```

### Ручная проверка

```python
# Проверка одной реплики/шарда
is_healthy = replication_manager.check_replica_health("replica_1")
is_healthy = sharding_manager.check_shard_health("shard_1")

# Проверка всех
health_status = replication_manager.check_all_replicas_health()
health_status = sharding_manager.check_all_shards_health()
```

## Динамическое управление

### Добавление реплики/шарда

Реплики и шарды можно добавлять динамически без перезапуска системы:

```python
# Добавление реплики
replication_manager.add_replica("new_replica", "http://localhost:8004")

# Добавление шарда
sharding_manager.add_shard("new_shard", "http://localhost:8005")
```

### Удаление реплики/шарда

```python
# Удаление реплики
replication_manager.remove_replica("old_replica")

# Удаление шарда (нельзя удалить последний шард)
sharding_manager.remove_shard("old_shard")
```

## Обработка ошибок

### Репликация

- Если реплика недоступна, операция записи все равно выполняется на первичном хранилище
- Ошибки репликации логируются, но не прерывают выполнение
- Недоступные реплики помечаются как нездоровые и исключаются из репликации

### Шардирование

- Если шард недоступен, он исключается из распределения
- Поиск выполняется только на доступных шардах
- При добавлении нового шарда он автоматически включается в распределение

## Примеры

Полные примеры использования доступны в файле `examples/replication_sharding_example.py`.

Запуск примера:

```bash
python examples/replication_sharding_example.py
```

## Архитектура

### Репликация

```
┌─────────────┐
│   Primary   │───┐
│  Storage    │   │
└─────────────┘   │
                  │ HTTP API
        ┌─────────┴─────────┐
        │                   │
┌───────▼──────┐   ┌────────▼──────┐
│  Replica 1  │   │  Replica 2   │
│  (8001)     │   │  (8002)       │
└─────────────┘   └───────────────┘
```

### Шардирование

```
        ┌──────────────┐
        │   Query      │
        │  Processor   │
        └──────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Shard 0│ │Shard 1│ │Shard 2│
└───────┘ └───────┘ └───────┘
```

## Ограничения и будущие улучшения

### Текущие ограничения

1. Полная синхронизация данных при добавлении реплики требует ручной реализации
2. Перераспределение данных между шардами - заглушка
3. Удаленные шарды в поиске используют первичное хранилище (требует HTTP запросов)

### Планируемые улучшения

- [ ] Полная синхронизация данных при добавлении реплики
- [ ] Автоматическое перераспределение данных между шардами
- [ ] Поддержка HTTP запросов для удаленных шардов в поиске
- [ ] Отдельные индексы для каждого шарда
- [ ] Консистентность данных (quorum reads/writes)

## Требования

Для использования репликации и шардирования требуется:

- Python 3.11+
- `requests` библиотека (автоматически устанавливается с проектом)

Установка зависимостей:

```bash
pip install -e .
```

или

```bash
poetry install
```

