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
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mlvectordb.api.main import app


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run MLVectorDB REST API server")
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
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
    
    uvicorn.run(
        "mlvectordb.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()