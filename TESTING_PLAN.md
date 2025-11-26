# План тестирования репликации и шардирования MLVectorDB

Этот документ содержит подробный пошаговый план запуска и тестирования реализаций репликации и шардирования.

## Предварительные требования

1. Python 3.11+
2. Установленные зависимости:
   ```bash
   pip install -e .
   # или
   poetry install
   ```

3. Терминалы/окна командной строки:
   - Минимум 3 терминала для тестирования репликации
   - Минимум 2 терминала для тестирования шардирования

## Часть 1: Тестирование репликации

### Шаг 1.1: Запуск первичного сервера

**Терминал 1:**
```bash
python -m src.mlvectordb.api.server --enable-replication --port 8000 --log-level info
```

Ожидаемый вывод:
```
Starting MLVectorDB API server on 127.0.0.1:8000
Replication: ENABLED
API documentation available at: http://127.0.0.1:8000/docs
```

### Шаг 1.2: Запуск реплик

**Терминал 2:**
```bash
python -m src.mlvectordb.api.server --enable-replication --port 8001 --log-level info
```

**Терминал 3:**
```bash
python -m src.mlvectordb.api.server --enable-replication --port 8002 --log-level info
```

### Шаг 1.3: Проверка работоспособности серверов

**Терминал 4 (для выполнения команд):**

```bash
# Проверка первичного сервера
curl http://localhost:8000/health

# Проверка реплик
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Ожидаемый результат: `{"status": "healthy"}` от всех серверов

### Шаг 1.4: Добавление реплик на первичном сервере

```bash
# Добавляем первую реплику
curl -X POST "http://localhost:8000/replication/replicas?replica_id=replica_1&replica_url=http://localhost:8001"

# Добавляем вторую реплику
curl -X POST "http://localhost:8000/replication/replicas?replica_id=replica_2&replica_url=http://localhost:8002"
```

Ожидаемый результат:
```json
{"status": "success", "message": "Реплика replica_1 добавлена"}
{"status": "success", "message": "Реплика replica_2 добавлена"}
```

### Шаг 1.5: Проверка информации о репликации

```bash
curl http://localhost:8000/replication/info
```

Ожидаемый результат: JSON с информацией о репликах, включая их статус

### Шаг 1.6: Вставка данных на первичный сервер

```bash
# Вставляем первый вектор
curl -X POST "http://localhost:8000/vectors?namespace=test" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1.0, 2.0, 3.0, 4.0],
    "metadata": {"test": "data1", "id": 1}
  }'

# Вставляем второй вектор
curl -X POST "http://localhost:8000/vectors?namespace=test" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [5.0, 6.0, 7.0, 8.0],
    "metadata": {"test": "data2", "id": 2}
  }'
```

### Шаг 1.7: Проверка репликации данных

```bash
# Проверяем, что данные есть на репликах
curl "http://localhost:8001/namespaces/vectors?namespace=test"
curl "http://localhost:8002/namespaces/vectors?namespace=test"
```

**Ожидаемый результат:** Векторы должны появиться на репликах (может потребоваться несколько секунд для асинхронной репликации)

### Шаг 1.8: Проверка здоровья реплик

```bash
# Проверка одной реплики
curl http://localhost:8000/replication/replicas/replica_1/health

# Проверка всех реплик
curl http://localhost:8000/replication/replicas/health
```

### Шаг 1.9: Тестирование синхронизации реплики

```bash
# Синхронизируем реплику (если она отстала)
curl -X POST "http://localhost:8000/replication/replicas/replica_1/sync"

# Синхронизация конкретного namespace
curl -X POST "http://localhost:8000/replication/replicas/replica_1/sync?namespace=test"
```

### Шаг 1.10: Тестирование удаления реплики

```bash
# Удаляем реплику
curl -X DELETE http://localhost:8000/replication/replicas/replica_2

# Проверяем, что реплика удалена
curl http://localhost:8000/replication/info
```

### Шаг 1.11: Тестирование отказоустойчивости

1. Остановите одну из реплик (Ctrl+C в терминале 2 или 3)
2. Вставьте новый вектор на первичный сервер
3. Проверьте, что репликация продолжает работать для доступных реплик
4. Запустите реплику снова
5. Выполните синхронизацию

## Часть 2: Тестирование шардирования

### Шаг 2.1: Запуск сервера с шардированием

**Терминал 1:**
```bash
python -m src.mlvectordb.api.server --enable-sharding --port 8000 --log-level info
```

Ожидаемый вывод:
```
Starting MLVectorDB API server on 127.0.0.1:8000
Sharding: ENABLED
```

### Шаг 2.2: Проверка информации о шардировании

```bash
curl http://localhost:8000/sharding/info
```

Ожидаемый результат: JSON с информацией о шардах (должно быть 2 локальных шарда по умолчанию)

### Шаг 2.3: Вставка данных для тестирования распределения

```bash
# Вставляем несколько векторов
for i in {1..10}; do
  curl -X POST "http://localhost:8000/vectors?namespace=test" \
    -H "Content-Type: application/json" \
    -d "{
      \"values\": [$i.0, $((i+1)).0, $((i+2)).0, $((i+3)).0],
      \"metadata\": {\"id\": $i, \"shard_test\": true}
    }"
  echo ""
done
```

### Шаг 2.4: Проверка распределения данных

```bash
# Получаем все векторы
curl "http://localhost:8000/namespaces/vectors?namespace=test"

# Проверяем информацию о шардах
curl http://localhost:8000/sharding/info
```

### Шаг 2.5: Добавление нового шарда

```bash
# Добавляем локальный шард (без URL)
curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_2"

# Или добавляем удаленный шард (если есть другой сервер)
# curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_remote&shard_url=http://localhost:8001"
```

### Шаг 2.6: Проверка здоровья шардов

```bash
# Проверка одного шарда
curl http://localhost:8000/sharding/shards/shard_0/health

# Проверка всех шардов
curl http://localhost:8000/sharding/shards/health
```

### Шаг 2.7: Тестирование поиска с шардированием

```bash
# Выполняем поиск
curl -X POST "http://localhost:8000/search?namespace=test" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [5.0, 6.0, 7.0, 8.0],
    "top_k": 5,
    "metric": "cosine"
  }'
```

**Ожидаемый результат:** Результаты должны быть агрегированы из всех шардов

### Шаг 2.8: Тестирование перераспределения данных

```bash
# Перераспределяем данные между шардами
curl -X POST "http://localhost:8000/sharding/shards/shard_0/redistribute/shard_1"

# Проверяем результат
curl http://localhost:8000/sharding/info
```

### Шаг 2.9: Удаление шарда

```bash
# Удаляем шард (нельзя удалить последний)
curl -X DELETE http://localhost:8000/sharding/shards/shard_2

# Проверяем информацию
curl http://localhost:8000/sharding/info
```

## Часть 3: Комбинированное тестирование (Репликация + Шардирование)

### Шаг 3.1: Запуск сервера с обоими механизмами

**Терминал 1:**
```bash
python -m src.mlvectordb.api.server --enable-replication --enable-sharding --port 8000
```

### Шаг 3.2: Настройка реплик

```bash
# Запустите реплики на портах 8001 и 8002 (как в части 1)
# Затем добавьте их:
curl -X POST "http://localhost:8000/replication/replicas?replica_id=replica_1&replica_url=http://localhost:8001"
curl -X POST "http://localhost:8000/replication/replicas?replica_id=replica_2&replica_url=http://localhost:8002"
```

### Шаг 3.3: Комплексное тестирование

```bash
# 1. Вставляем данные
curl -X POST "http://localhost:8000/vectors?namespace=combined" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1.0, 2.0, 3.0],
    "metadata": {"test": "combined"}
  }'

# 2. Проверяем информацию о системе
curl http://localhost:8000/storage/info

# 3. Выполняем поиск
curl -X POST "http://localhost:8000/search?namespace=combined" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [1.0, 2.0, 3.0],
    "top_k": 5
  }'

# 4. Проверяем репликацию
curl http://localhost:8000/replication/info

# 5. Проверяем шардирование
curl http://localhost:8000/sharding/info
```

## Часть 4: Автоматизированное тестирование

### Создание тестового скрипта

Создайте файл `test_replication_sharding.py`:

```python
#!/usr/bin/env python3
"""Автоматизированные тесты для репликации и шардирования."""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Тест проверки здоровья."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✓ Health check passed")

def test_replication_setup():
    """Тест настройки репликации."""
    # Добавляем реплику
    response = requests.post(
        f"{BASE_URL}/replication/replicas",
        params={"replica_id": "test_replica", "replica_url": "http://localhost:8001"}
    )
    print(f"✓ Replica added: {response.json()}")

def test_insert_and_replicate():
    """Тест вставки и репликации."""
    vector = {
        "values": [1.0, 2.0, 3.0, 4.0],
        "metadata": {"test": "replication"}
    }
    response = requests.post(
        f"{BASE_URL}/vectors?namespace=test",
        json=vector
    )
    assert response.status_code == 201
    print("✓ Vector inserted")

def test_sharding_info():
    """Тест информации о шардировании."""
    response = requests.get(f"{BASE_URL}/sharding/info")
    assert response.status_code == 200
    info = response.json()
    print(f"✓ Sharding info: {info['total_shards']} shards, {info['healthy_shards']} healthy")

def main():
    """Запуск всех тестов."""
    print("Running automated tests...")
    print("=" * 50)
    
    try:
        test_health()
        test_replication_setup()
        test_insert_and_replicate()
        test_sharding_info()
        
        print("=" * 50)
        print("All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

Запуск:
```bash
python test_replication_sharding.py
```

## Часть 5: Тестирование производительности

### Нагрузочное тестирование

```bash
# Вставка большого количества векторов
for i in {1..1000}; do
  curl -X POST "http://localhost:8000/vectors?namespace=perf" \
    -H "Content-Type: application/json" \
    -d "{
      \"values\": [$i.0, $((i+1)).0, $((i+2)).0],
      \"metadata\": {\"id\": $i}
    }" &
done
wait

# Проверка времени поиска
time curl -X POST "http://localhost:8000/search?namespace=perf" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [500.0, 501.0, 502.0],
    "top_k": 10
  }'
```

## Часть 6: Тестирование отказоустойчивости

### Тест 6.1: Падение реплики

1. Запустите систему с репликацией
2. Добавьте реплики
3. Вставьте данные
4. Остановите одну реплику (Ctrl+C)
5. Вставьте новые данные
6. Проверьте, что система продолжает работать
7. Запустите реплику снова
8. Выполните синхронизацию

### Тест 6.2: Падение шарда

1. Запустите систему с шардированием
2. Вставьте данные
3. Остановите один из шардов (если удаленный)
4. Выполните поиск - должен работать с доступными шардами
5. Запустите шард снова
6. Проверьте здоровье

## Часть 7: Визуализация и мониторинг

### Использование Swagger UI

Откройте в браузере:
- http://localhost:8000/docs - интерактивная документация API
- http://localhost:8000/redoc - альтернативная документация

### Мониторинг через API

```bash
# Регулярная проверка статуса
watch -n 5 'curl -s http://localhost:8000/replication/info | python -m json.tool'
watch -n 5 'curl -s http://localhost:8000/sharding/info | python -m json.tool'
```

## Часть 8: Демонстрационный сценарий

### Полный сценарий демонстрации

1. **Запуск инфраструктуры:**
   ```bash
   # Терминал 1: Первичный сервер
   python -m src.mlvectordb.api.server --enable-replication --enable-sharding --port 8000
   
   # Терминал 2: Реплика 1
   python -m src.mlvectordb.api.server --enable-replication --port 8001
   
   # Терминал 3: Реплика 2
   python -m src.mlvectordb.api.server --enable-replication --port 8002
   ```

2. **Настройка:**
   ```bash
   # Добавляем реплики
   curl -X POST "http://localhost:8000/replication/replicas?replica_id=r1&replica_url=http://localhost:8001"
   curl -X POST "http://localhost:8000/replication/replicas?replica_id=r2&replica_url=http://localhost:8002"
   ```

3. **Вставка данных:**
   ```bash
   for i in {1..20}; do
     curl -X POST "http://localhost:8000/vectors?namespace=demo" \
       -H "Content-Type: application/json" \
       -d "{\"values\": [$i.0, $((i+1)).0, $((i+2)).0], \"metadata\": {\"id\": $i}}"
   done
   ```

4. **Проверка репликации:**
   ```bash
   curl "http://localhost:8001/namespaces/vectors?namespace=demo" | python -m json.tool | head -20
   curl "http://localhost:8002/namespaces/vectors?namespace=demo" | python -m json.tool | head -20
   ```

5. **Проверка шардирования:**
   ```bash
   curl http://localhost:8000/sharding/info | python -m json.tool
   ```

6. **Поиск:**
   ```bash
   curl -X POST "http://localhost:8000/search?namespace=demo" \
     -H "Content-Type: application/json" \
     -d '{"query": [10.0, 11.0, 12.0], "top_k": 5}' | python -m json.tool
   ```

7. **Динамическое управление:**
   ```bash
   # Добавляем новый шард
   curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_new"
   
   # Проверяем здоровье
   curl http://localhost:8000/replication/replicas/health
   curl http://localhost:8000/sharding/shards/health
   ```

## Устранение неполадок

### Проблема: Реплика не получает данные

**Решение:**
1. Проверьте, что реплика запущена и доступна
2. Проверьте здоровье: `curl http://localhost:8000/replication/replicas/{id}/health`
3. Выполните синхронизацию: `curl -X POST http://localhost:8000/replication/replicas/{id}/sync`

### Проблема: Шард недоступен

**Решение:**
1. Проверьте здоровье: `curl http://localhost:8000/sharding/shards/{id}/health`
2. Убедитесь, что шард запущен (если удаленный)
3. Проверьте URL шарда в конфигурации

### Проблема: Медленная репликация

**Решение:**
1. Проверьте сетевую задержку между серверами
2. Увеличьте таймаут в настройках ReplicationManager
3. Проверьте логи на наличие ошибок

## Заключение

Этот план покрывает все основные сценарии использования репликации и шардирования. Для полной демонстрации рекомендуется выполнить все части последовательно.

