# QuantRS2 Docker Images

This directory contains Docker configurations for building and deploying QuantRS2 in various environments.

## Available Images

### Base Image (`quantrs2:latest`)
- **File**: `Dockerfile`
- **Purpose**: Production-ready QuantRS2 installation
- **Size**: ~500MB (optimized)
- **Use Cases**: Production deployments, CI/CD, minimal installations

```bash
docker run -it quantrs2:latest python -c "import quantrs2; print('QuantRS2 ready!')"
```

### Development Image (`quantrs2:dev`)
- **File**: `Dockerfile.dev`
- **Purpose**: Full development environment
- **Size**: ~2GB (includes all dev tools)
- **Use Cases**: Development, debugging, testing
- **Includes**: pytest, black, mypy, Rust tools, debugging tools

```bash
docker run -it quantrs2:dev bash
```

### Jupyter Image (`quantrs2:jupyter`)
- **File**: `Dockerfile.jupyter`
- **Purpose**: Interactive quantum computing with Jupyter Lab
- **Size**: ~1.5GB
- **Use Cases**: Research, tutorials, interactive development
- **Includes**: JupyterLab, quantum libraries, visualization tools

```bash
docker run -p 8888:8888 quantrs2:jupyter
# Open http://localhost:8888 in your browser
```

### GPU Image (`quantrs2:gpu`)
- **File**: `Dockerfile.gpu`
- **Purpose**: GPU-accelerated quantum simulations
- **Size**: ~3GB (includes CUDA)
- **Use Cases**: High-performance simulations, large quantum circuits
- **Requirements**: NVIDIA Docker, CUDA-compatible GPU

```bash
docker run --gpus all quantrs2:gpu
```

## Quick Start

### 1. Build All Images
```bash
# Build all images with default settings
./docker/build.sh

# Build with custom registry and push
./docker/build.sh -r myregistry/quantrs2 -p

# Build in parallel for faster builds
./docker/build.sh -j

# Build and test
./docker/build.sh -t
```

### 2. Run with Docker Compose
```bash
# Start full development environment
docker-compose up

# Start only Jupyter service
docker-compose up quantrs2-jupyter

# Start with GPU support
docker-compose up quantrs2-gpu
```

### 3. Individual Container Usage

#### Quick Test
```bash
docker run --rm quantrs2:latest python -c "
import quantrs2
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)
result = circuit.run()
print('Bell state created!')
"
```

#### Interactive Development
```bash
docker run -it --rm \
  -v $(pwd):/app \
  -p 8888:8888 \
  quantrs2:dev bash
```

#### Jupyter Lab Session
```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/quantrs/notebooks \
  quantrs2:jupyter
```

#### GPU-Accelerated Simulation
```bash
docker run --gpus all --rm \
  quantrs2:gpu python -c "
import quantrs2
# Large quantum circuit simulation
circuit = quantrs2.Circuit(20)
# ... your quantum algorithm here
"
```

## Docker Compose Services

The `docker-compose.yml` file defines a complete QuantRS2 ecosystem:

### Core Services
- **quantrs2-base**: Base QuantRS2 service
- **quantrs2-jupyter**: Jupyter Lab interface (port 8888)
- **quantrs2-gpu**: GPU-accelerated service
- **quantrs2-dev**: Development environment

### Supporting Services
- **quantrs2-db**: PostgreSQL database for experiments
- **quantrs2-redis**: Redis for caching and sessions
- **reverse-proxy**: Traefik reverse proxy
- **quantrs2-docs**: Documentation server (port 8000)

### Monitoring
- **quantrs2-prometheus**: Metrics collection (port 9090)
- **quantrs2-grafana**: Visualization dashboard (port 3001)

### Usage
```bash
# Start all services
docker-compose up

# Start specific services
docker-compose up quantrs2-jupyter quantrs2-db

# View logs
docker-compose logs -f quantrs2-jupyter

# Scale services
docker-compose up --scale quantrs2-base=3

# Stop all services
docker-compose down
```

## Environment Variables

### Common Variables
- `QUANTRS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `QUANTRS_GPU_ENABLED`: Enable GPU acceleration (true/false)
- `QUANTRS_ENV`: Environment (development, production)

### Jupyter Variables
- `JUPYTER_ENABLE_LAB`: Enable JupyterLab (default: yes)
- `JUPYTER_TOKEN`: Security token (empty for no authentication)

### Database Variables
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password

## Volume Mounts

### Persistent Data
- `quantrs2-data`: Application data
- `jupyter-notebooks`: Jupyter notebooks
- `benchmark-results`: Performance benchmark results
- `postgres-data`: Database data
- `redis-data`: Redis cache data

### Development Mounts
```bash
# Mount source code for development
-v $(pwd):/app

# Mount notebooks directory
-v $(pwd)/notebooks:/home/quantrs/notebooks

# Mount data directory
-v $(pwd)/data:/app/data
```

## Network Configuration

The Docker Compose setup creates a custom network (`quantrs2-network`) with:
- **Subnet**: 172.20.0.0/16
- **Driver**: bridge
- **Service Discovery**: Automatic DNS resolution between services

### Service Access
- **Jupyter**: http://localhost:8888 or http://jupyter.quantrs2.local
- **Documentation**: http://localhost:8000 or http://docs.quantrs2.local
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090
- **Traefik Dashboard**: http://localhost:8081

## Security Considerations

### Production Deployment
1. **Change Default Passwords**: Update all default passwords in production
2. **Enable Authentication**: Configure proper authentication for Jupyter and other services
3. **Network Security**: Use secure networks and firewalls
4. **Resource Limits**: Set appropriate CPU and memory limits
5. **User Permissions**: Run containers with non-root users (already configured)

### Development Security
```bash
# Run with security options
docker run --security-opt=no-new-privileges --read-only quantrs2:latest

# Limit resources
docker run --memory=1g --cpus=1.0 quantrs2:latest
```

## Performance Optimization

### Build Optimization
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with cache mount
docker build --mount=type=cache,target=/root/.cargo ...

# Multi-stage builds reduce image size
# (already implemented in Dockerfiles)
```

### Runtime Optimization
```bash
# Use specific CPU and memory limits
docker run --memory=2g --cpus=2.0 quantrs2:latest

# For GPU workloads
docker run --gpus all --shm-size=1g quantrs2:gpu
```

## Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -f docker/Dockerfile .

# Check Docker version and available space
docker version
df -h
```

#### 2. Permission Issues
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) ./data

# Run as current user
docker run --user $(id -u):$(id -g) quantrs2:latest
```

#### 3. GPU Issues
```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify GPU availability in container
docker run --gpus all quantrs2:gpu nvidia-smi
```

#### 4. Network Issues
```bash
# Check container networking
docker network ls
docker network inspect quantrs2_quantrs2-network

# Test service connectivity
docker exec quantrs2-jupyter ping quantrs2-db
```

### Debugging Containers
```bash
# Get shell access to running container
docker exec -it quantrs2-jupyter bash

# View container logs
docker logs --tail 50 -f quantrs2-jupyter

# Inspect container configuration
docker inspect quantrs2-jupyter

# Check resource usage
docker stats quantrs2-jupyter
```

## Advanced Usage

### Custom Builds
```bash
# Build with custom Python version
docker build --build-arg PYTHON_VERSION=3.10 -f docker/Dockerfile .

# Build with additional packages
docker build --build-arg EXTRA_PACKAGES="tensorflow pytorch" -f docker/Dockerfile .
```

### Multi-Architecture Builds
```bash
# Enable buildx
docker buildx create --use

# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t quantrs2:latest .
```

### Registry Operations
```bash
# Tag for different registries
docker tag quantrs2:latest myregistry.com/quantrs2:v1.0.0

# Push to registry
docker push myregistry.com/quantrs2:v1.0.0

# Pull from registry
docker pull myregistry.com/quantrs2:v1.0.0
```

## Monitoring and Logging

### Container Logs
```bash
# View logs from all services
docker-compose logs

# Follow logs from specific service
docker-compose logs -f quantrs2-jupyter

# View logs with timestamps
docker-compose logs -t quantrs2-base
```

### Performance Monitoring
Access Grafana at http://localhost:3001 (admin/admin) to view:
- Container resource usage
- Quantum algorithm performance metrics
- System health dashboards
- Custom QuantRS2 metrics

### Health Checks
All images include health checks:
```bash
# Check container health
docker ps  # Shows health status

# Manual health check
docker exec quantrs2-base python -c "import quantrs2; print('healthy')"
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Build and Test Docker Images
on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build images
        run: ./docker/build.sh -t
      - name: Run tests
        run: docker run --rm quantrs2:latest python -m pytest
```

### GitLab CI Example
```yaml
docker-build:
  stage: build
  script:
    - ./docker/build.sh -r $CI_REGISTRY_IMAGE -p
  only:
    - main
```

## Contributing

When contributing to Docker configurations:

1. **Test all images**: Ensure all Dockerfiles build successfully
2. **Update documentation**: Keep this README updated with changes
3. **Follow best practices**: Use multi-stage builds, minimize layers
4. **Security review**: Ensure no secrets or sensitive data in images
5. **Performance testing**: Verify images perform well in target environments

### Development Workflow
```bash
# 1. Make changes to Dockerfiles
# 2. Test locally
./docker/build.sh -t

# 3. Test with compose
docker-compose up --build

# 4. Run integration tests
docker run --rm quantrs2:dev python -m pytest tests/

# 5. Submit PR with updated documentation
```

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)

For QuantRS2-specific questions, see the main project documentation or open an issue on the GitHub repository.