# CADET

CADET is a code evolution program inspired by the "alpha evolve" paper. This project is actively evolving, and we welcome contributions and forks!

## Features

- Code evolution through genetic programming
- Docker container support
- Configurable evaluation methods

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized setup)
- Git (for version control)

## Quick Start

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/DemenYunn/CADET.git
   cd CADET
   ```
2.Install dependencies:
```pip install -r requirements.txt```
Run the program:
```python main.py```

# Docker Setup

# Build the Docker image:

```docker-compose build```
```
# Run the container:
docker-compose run --rm code-evolver python main.py
```
Configuration
Environment Variables
Create a .env file in the project root to set environment variables:
```
env
```

```PYTHONUNBUFFERED=1```

# Add other environment variables here
Docker Configuration
Edit docker-compose.yml to:

# Mount additional volumes
# Expose ports
# Set environment variables

# Copying a file over to docker container
This is needed because it will ask for a file and if you dont have it in your os/docker container/environment. So this command is only meant for people to run if you are running as a docker container we set up earlier in the same directory you downloaded from github "CADET" or "CADET-main" either way its got the python file in it too and a docker file ready for you to build and run. But before you run you must copy and paste this command or just type it. Make sure you change the file names if needed
```
docker cp to_get_better.py {container_name}:/app/to_get_better.py
```
# Finding the name of container
code-evolver:
    build: .
    container_name: code-evolver
Is how it looks in the file so you can replace {container-name} with code-evolver
or whatever is in your configuration ending in .yml.
```
docker-compose.yml`
```


## Development
For development, set ENVIRONMENT=development in your .env file to install development dependencies.

## Contributing
Contributions are welcome! Please submit a pull request or create an issue to discuss your ideas.

## License
MIT

## Acknowledgments
Inspired by the "alpha evolve" paper
Built with Python and Docker
