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
    
    args = parser.parse_args()
    
    print(f"Starting MLVectorDB API server on {args.host}:{args.port}")
    print(f"API documentation available at: http://{args.host}:{args.port}/docs")
    print(f"Interactive API docs at: http://{args.host}:{args.port}/redoc")


    from src.mlvectordb.implementations.query_processor import QueryProcessor
    from src.mlvectordb import StorageEngineInMemory
    from src.mlvectordb.implementations.index import Index
    from src.mlvectordb.api.rest_api import RestAPI

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
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=None
    )


if __name__ == "__main__":
    main()
