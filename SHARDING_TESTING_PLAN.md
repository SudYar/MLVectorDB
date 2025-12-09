# План тестирования и демонстрации шардирования MLVectorDB

Этот документ содержит подробный поэтапный план тестирования исправленной реализации шардирования.

## Исправленные проблемы

1. ✅ Векторы теперь добавляются ТОЛЬКО в шарды, без дублирования в первичное хранилище
2. ✅ Улучшен health check для реплик - добавлена повторная проверка при добавлении
3. ✅ Исправлен health check для локальных шардов - проверяется наличие StorageEngine
4. ✅ Добавлен метод `add_local_shard()` для динамического добавления локальных шардов
5. ✅ Добавлен REST API endpoint `/sharding/shards/local` для добавления локальных шардов
6. ✅ Добавлен метод `write_batch_to_remote_shard()` для эффективной batch записи

## Предварительные требования

1. Python 3.11+
2. Установленные зависимости:
   ```bash
   pip install -e .
   ```
3. Минимум 2 терминала для тестирования

## Этап 1: Базовое тестирование шардирования

### Шаг 1.1: Запуск сервера с шардированием

**Терминал 1:**
```bash
python -m src.mlvectordb.api.server --enable-sharding --port 8000 --log-level info
```

Ожидаемый вывод:
```
Starting MLVectorDB API server on 127.0.0.1:8000
Sharding: ENABLED
Sharding manager initialized with 2 local shards
```

### Шаг 1.2: Проверка информации о шардировании

**Терминал 2:**
```bash
curl http://localhost:8000/sharding/info | python -m json.tool
```

Ожидаемый результат:
```json
{
  "total_shards": 2,
  "healthy_shards": 2,
  "sharding_strategy": "hash",
  "shards": {
    "shard_0": {
      "is_healthy": true,
      "is_local": true,
      "vector_count": 0
    },
    "shard_1": {
      "is_healthy": true,
      "is_local": true,
      "vector_count": 0
    }
  }
}
```

### Шаг 1.3: Вставка векторов и проверка распределения

```bash
# Вставляем 10 векторов
for i in {1..10}; do
  curl -X POST "http://localhost:8000/vectors?namespace=test" \
    -H "Content-Type: application/json" \
    -d "{
      \"values\": [$i.0, $((i+1)).0, $((i+2)).0, $((i+3)).0],
      \"metadata\": {\"id\": $i, \"test\": \"sharding\"}
    }"
  echo ""
done
```

### Шаг 1.4: Проверка распределения по шардам

```bash
# Проверяем общее количество векторов
curl "http://localhost:8000/namespaces/vectors?namespace=test" | python -m json.tool | grep -c "\"id\""

# Проверяем информацию о шардах
curl http://localhost:8000/sharding/info | python -m json.tool
```

**Ожидаемый результат:** Векторы должны быть распределены между шардами (не все в одном шарде).

### Шаг 1.5: Проверка, что векторы НЕ дублируются в первичном хранилище

```bash
# Получаем информацию о хранилище
curl http://localhost:8000/storage/info | python -m json.tool
```

**Ожидаемый результат:** При шардировании векторы должны быть ТОЛЬКО в шардах, не в первичном хранилище (кроме fallback случаев).

## Этап 2: Тестирование динамического добавления шардов

### Шаг 2.1: Добавление локального шарда

```bash
# Добавляем новый локальный шард
curl -X POST "http://localhost:8000/sharding/shards/local?shard_id=shard_2"
```

Ожидаемый результат:
```json
{
  "status": "success",
  "message": "Локальный шард shard_2 добавлен",
  "shard_id": "shard_2",
  "is_local": true
}
```

### Шаг 2.2: Проверка информации о шардах после добавления

```bash
curl http://localhost:8000/sharding/info | python -m json.tool
```

**Ожидаемый результат:** Должно быть 3 шарда, все здоровые.

### Шаг 2.3: Вставка новых векторов и проверка распределения

```bash
# Вставляем еще 5 векторов
for i in {11..15}; do
  curl -X POST "http://localhost:8000/vectors?namespace=test" \
    -H "Content-Type: application/json" \
    -d "{
      \"values\": [$i.0, $((i+1)).0, $((i+2)).0, $((i+3)).0],
      \"metadata\": {\"id\": $i, \"test\": \"new_shard\"}
    }"
  echo ""
done
```

**Ожидаемый результат:** Новые векторы должны распределяться между всеми тремя шардами.

### Шаг 2.4: Добавление удаленного шарда (если есть другой сервер)

Если у вас есть другой сервер на порту 8001:
```bash
# Запустите второй сервер в другом терминале:
# python -m src.mlvectordb.api.server --port 8001

# Затем добавьте его как удаленный шард:
curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_remote&shard_url=http://localhost:8001"
```

## Этап 3: Тестирование поиска с шардированием

### Шаг 3.1: Выполнение поиска

```bash
curl -X POST "http://localhost:8000/search?namespace=test" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [5.0, 6.0, 7.0, 8.0],
    "top_k": 5,
    "metric": "cosine"
  }' | python -m json.tool
```

**Ожидаемый результат:** Должны вернуться результаты из всех шардов, отсортированные по score.

### Шаг 3.2: Проверка корректности результатов

Результаты должны содержать:
- `id` - UUID вектора
- `values` - значения вектора
- `metadata` - метаданные
- `score` - оценка похожести

## Этап 4: Тестирование health check

### Шаг 4.1: Проверка здоровья всех шардов

```bash
curl http://localhost:8000/sharding/shards/health | python -m json.tool
```

Ожидаемый результат:
```json
{
  "shards_health": {
    "shard_0": true,
    "shard_1": true,
    "shard_2": true
  }
}
```

### Шаг 4.2: Проверка здоровья конкретного шарда

```bash
curl http://localhost:8000/sharding/shards/shard_0/health | python -m json.tool
```

### Шаг 4.3: Тестирование добавления шарда без StorageEngine

```bash
# Добавляем шард без StorageEngine (только ID)
curl -X POST "http://localhost:8000/sharding/shards?shard_id=shard_empty"

# Проверяем его здоровье
curl http://localhost:8000/sharding/shards/shard_empty/health | python -m json.tool
```

**Ожидаемый результат:** Шард должен быть помечен как нездоровый (`is_healthy: false`).

## Этап 5: Тестирование репликации с шардированием

### Шаг 5.1: Запуск сервера с репликацией и шардированием

**Терминал 1 (Primary):**
```bash
python -m src.mlvectordb.api.server --enable-replication primary --enable-sharding --port 8000
```

**Терминал 2 (Replica 1):**
```bash
python -m src.mlvectordb.api.server --enable-replication replica --replica-name replica_1 --primary-url http://localhost:8000 --port 8001
```

**Терминал 3 (Replica 2):**
```bash
python -m src.mlvectordb.api.server --enable-replication replica --replica-name replica_2 --primary-url http://localhost:8000 --port 8002
```

### Шаг 5.2: Проверка добавления реплик

```bash
# Проверяем информацию о репликации
curl http://localhost:8000/replication/info | python -m json.tool
```

**Ожидаемый результат:** Реплики должны быть добавлены и иметь статус `is_healthy: true` (после повторных проверок).

### Шаг 5.3: Вставка данных и проверка репликации

```bash
# Вставляем вектор на primary
curl -X POST "http://localhost:8000/vectors?namespace=repl_test" \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1.0, 2.0, 3.0, 4.0],
    "metadata": {"test": "replication"}
  }'

# Проверяем на репликах (может потребоваться несколько секунд)
sleep 2
curl "http://localhost:8001/namespaces/vectors?namespace=repl_test" | python -m json.tool
curl "http://localhost:8002/namespaces/vectors?namespace=repl_test" | python -m json.tool
```

**Ожидаемый результат:** Вектор должен появиться на репликах.

## Этап 6: Тестирование отказоустойчивости

### Шаг 6.1: Падение и восстановление шарда

1. Добавьте удаленный шард (если есть)
2. Остановите сервер удаленного шарда
3. Проверьте health check - шард должен быть помечен как нездоровый
4. Вставьте новые векторы - система должна работать с доступными шардами
5. Запустите сервер удаленного шарда снова
6. Проверьте health check - шард должен стать здоровым

### Шаг 6.2: Падение и восстановление реплики

1. Остановите одну из реплик
2. Вставьте новые векторы на primary
3. Проверьте, что репликация продолжает работать для доступных реплик
4. Запустите реплику снова
5. Выполните синхронизацию:
   ```bash
   curl -X POST "http://localhost:8000/replication/replicas/replica_1/sync"
   ```

## Этап 7: Комплексное тестирование

### Шаг 7.1: Массовая вставка с проверкой распределения

```bash
# Вставляем 100 векторов
for i in {1..100}; do
  curl -X POST "http://localhost:8000/vectors?namespace=stress" \
    -H "Content-Type: application/json" \
    -d "{
      \"values\": [$i.0, $((i+1)).0, $((i+2)).0, $((i+3)).0],
      \"metadata\": {\"id\": $i}
    }" > /dev/null 2>&1
done

# Проверяем распределение
curl http://localhost:8000/sharding/info | python -m json.tool
```

**Ожидаемый результат:** Векторы должны быть равномерно распределены между шардами.

### Шаг 7.2: Поиск по большому количеству данных

```bash
curl -X POST "http://localhost:8000/search?namespace=stress" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [50.0, 51.0, 52.0, 53.0],
    "top_k": 10,
    "metric": "cosine"
  }' | python -m json.tool
```

## Этап 8: Демонстрационный сценарий

### Полный сценарий демонстрации

1. **Запуск инфраструктуры:**
   ```bash
   # Терминал 1: Primary с шардированием и репликацией
   python -m src.mlvectordb.api.server --enable-replication primary --enable-sharding --port 8000
   
   # Терминал 2: Replica 1
   python -m src.mlvectordb.api.server --enable-replication replica --replica-name r1 --primary-url http://localhost:8000 --port 8001
   
   # Терминал 3: Replica 2
   python -m src.mlvectordb.api.server --enable-replication replica --replica-name r2 --primary-url http://localhost:8000 --port 8002
   ```

2. **Проверка начального состояния:**
   ```bash
   curl http://localhost:8000/sharding/info | python -m json.tool
   curl http://localhost:8000/replication/info | python -m json.tool
   ```

3. **Вставка данных:**
   ```bash
   for i in {1..20}; do
     curl -X POST "http://localhost:8000/vectors?namespace=demo" \
       -H "Content-Type: application/json" \
       -d "{\"values\": [$i.0, $((i+1)).0, $((i+2)).0], \"metadata\": {\"id\": $i}}"
   done
   ```

4. **Добавление нового локального шарда:**
   ```bash
   curl -X POST "http://localhost:8000/sharding/shards/local?shard_id=shard_new"
   ```

5. **Проверка распределения:**
   ```bash
   curl http://localhost:8000/sharding/info | python -m json.tool
   ```

6. **Поиск:**
   ```bash
   curl -X POST "http://localhost:8000/search?namespace=demo" \
     -H "Content-Type: application/json" \
     -d '{"query": [10.0, 11.0, 12.0], "top_k": 5}' | python -m json.tool
   ```

7. **Проверка здоровья:**
   ```bash
   curl http://localhost:8000/sharding/shards/health | python -m json.tool
   curl http://localhost:8000/replication/replicas/health | python -m json.tool
   ```

## Критерии успешного тестирования

✅ **Векторы добавляются только в шарды** - нет дублирования в первичное хранилище  
✅ **Равномерное распределение** - векторы распределяются между шардами  
✅ **Динамическое добавление шардов** - можно добавлять шарды без перезапуска  
✅ **Health check работает** - корректно определяет здоровые/нездоровые шарды и реплики  
✅ **Поиск работает** - результаты агрегируются из всех шардов  
✅ **Репликация работает** - данные реплицируются на реплики  
✅ **Отказоустойчивость** - система продолжает работать при падении шарда/реплики  

## Устранение неполадок

### Проблема: Векторы не распределяются по шардам

**Решение:**
1. Проверьте, что шардирование включено: `curl http://localhost:8000/sharding/info`
2. Убедитесь, что есть минимум 2 здоровых шарда
3. Проверьте логи сервера на наличие ошибок

### Проблема: Health check показывает false для новой реплики

**Решение:**
1. Убедитесь, что реплика запущена и доступна
2. Подождите несколько секунд - health check выполняется в фоне
3. Проверьте вручную: `curl http://localhost:8000/replication/replicas/{id}/health`
4. Если реплика только что запущена, health check может обновиться через несколько секунд

### Проблема: Локальный шард показывает is_healthy: false

**Решение:**
1. Используйте endpoint `/sharding/shards/local` для добавления локального шарда с StorageEngine
2. Или добавьте StorageEngine через `add_local_shard()` в коде

## Заключение

Этот план покрывает все основные сценарии использования исправленного шардирования. Для полной демонстрации рекомендуется выполнить все этапы последовательно.

