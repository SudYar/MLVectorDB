#!/usr/bin/env python3
"""
CLI script to run the MLVectorDB REST API server.

Usage:
    python -m mlvectordb.api.server
    or
    python scripts/run_api.py
"""
import uvicorn
import argparse
import requests


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run MLVectorDB REST API server")
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--enable-replication",
        choices=["primary", "replica"],
        help="Choose replication support"
    )
    parser.add_argument(
        "--primary-url",
        default="http://localhost:8000/replication/replicas",
        help="Send primary url"
    )
    parser.add_argument(
        "--replica-name",
        help="Name this replica"
    )
    parser.add_argument(
        "--enable-sharding",
        action="store_true",
        help="Enable sharding support"
    )
    
    args = parser.parse_args()
    
    print(f"Starting MLVectorDB API server on {args.host}:{args.port}")
    print(f"API documentation available at: http://{args.host}:{args.port}/docs")
    print(f"Interactive API docs at: http://{args.host}:{args.port}/redoc")
    if args.enable_replication:
        print("Replication: ENABLED")
    if args.enable_sharding:
        print("Sharding: ENABLED")

    from src.mlvectordb.implementations.query_processor import QueryProcessor
    from src.mlvectordb.implementations.query_processor_with_replication import QueryProcessorWithReplication
    from src.mlvectordb import StorageEngineInMemory
    from src.mlvectordb.implementations.index import Index
    from src.mlvectordb.api.rest_api import RestAPI

    # Инициализация компонентов
    primary_storage = StorageEngineInMemory()
    index = Index()
    
    # Настройка репликации и шардирования
    replication_manager = None
    sharding_manager = None
    
    if args.enable_replication or args.enable_sharding:
        from src.mlvectordb.implementations.replication_manager import ReplicationManagerImpl
        from src.mlvectordb.implementations.sharding_manager import ShardingManagerImpl
        
        if args.enable_replication:
            if "primary" == args.enable_replication:
                replication_manager = ReplicationManagerImpl(
                    primary_storage=primary_storage,
                    primary_replica_id="primary",
                    health_check_interval=5.0
                )
                print("Primary replication manager initialized")
            if "replica" == args.enable_replication:
                if args.replica_name and args.primary_url:
                    request = requests.post(url=f"{args.primary_url}/replication/replicas", params={
                        "replica_id": args.replica_name,
                        "replica_url": f"http://{args.host}:{args.port}"
                    })
        if args.enable_sharding:
            # Создаем локальные шарды
            shard_storages = {
                "shard_0": StorageEngineInMemory(),
                "shard_1": StorageEngineInMemory(),
            }
            sharding_manager = ShardingManagerImpl(
                shard_storages=shard_storages,
                sharding_strategy="hash",
                health_check_interval=5.0
            )
            print(f"Sharding manager initialized with {len(shard_storages)} local shards")

        qproc = QueryProcessorWithReplication(
            storage_engine=primary_storage,
            index=index,
            replication_manager=replication_manager,
            sharding_manager=sharding_manager
        )
    else:
        qproc = QueryProcessor(primary_storage, index)

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
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=None
    )


if __name__ == "__main__":
    main()
