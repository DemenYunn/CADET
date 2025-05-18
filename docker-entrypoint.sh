#!/bin/bash
set -e

# Initialize any required setup here
# For example, you might want to install development dependencies in development
if [ "$ENVIRONMENT" = "development" ]; then
    pip install -r requirements-dev.txt
    docker-compose run --rm code-evolver python code_evolver.py
fi

# Execute the command passed to the container
exec "$@"
