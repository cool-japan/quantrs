#!/bin/bash
# QuantRS2 Docker Build Script
# Builds all Docker images for the QuantRS2 ecosystem

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${DOCKER_REGISTRY:-"quantrs2"}
VERSION=${VERSION:-"latest"}
BUILD_ARGS=${BUILD_ARGS:-""}
PUSH=${PUSH:-false}
PARALLEL=${PARALLEL:-false}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
}

# Function to check if nvidia-docker is available (for GPU builds)
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_warning "NVIDIA Docker is not available. GPU builds will be skipped."
        return 1
    fi
    return 0
}

# Function to build a single image
build_image() {
    local dockerfile=$1
    local tag=$2
    local build_context=${3:-".."}
    local additional_args=${4:-""}
    
    print_status "Building image: ${REGISTRY}:${tag}"
    
    # Build the image
    if docker build \
        -f "docker/${dockerfile}" \
        -t "${REGISTRY}:${tag}" \
        -t "${REGISTRY}:${VERSION}-${tag}" \
        ${BUILD_ARGS} \
        ${additional_args} \
        "${build_context}"; then
        print_success "Successfully built ${REGISTRY}:${tag}"
        return 0
    else
        print_error "Failed to build ${REGISTRY}:${tag}"
        return 1
    fi
}

# Function to push image to registry
push_image() {
    local tag=$1
    
    if [ "$PUSH" = true ]; then
        print_status "Pushing image: ${REGISTRY}:${tag}"
        if docker push "${REGISTRY}:${tag}" && docker push "${REGISTRY}:${VERSION}-${tag}"; then
            print_success "Successfully pushed ${REGISTRY}:${tag}"
        else
            print_error "Failed to push ${REGISTRY}:${tag}"
            return 1
        fi
    fi
}

# Function to build all images sequentially
build_sequential() {
    local failed_builds=()
    
    # Build base image first
    if build_image "Dockerfile" "latest"; then
        push_image "latest"
    else
        failed_builds+=("base")
    fi
    
    # Build development image
    if build_image "Dockerfile.dev" "dev"; then
        push_image "dev"
    else
        failed_builds+=("dev")
    fi
    
    # Build Jupyter image (depends on base)
    if build_image "Dockerfile.jupyter" "jupyter"; then
        push_image "jupyter"
    else
        failed_builds+=("jupyter")
    fi
    
    # Build GPU image if NVIDIA Docker is available
    if check_nvidia_docker; then
        if build_image "Dockerfile.gpu" "gpu"; then
            push_image "gpu"
        else
            failed_builds+=("gpu")
        fi
    else
        print_warning "Skipping GPU image build (NVIDIA Docker not available)"
    fi
    
    # Report results
    if [ ${#failed_builds[@]} -eq 0 ]; then
        print_success "All images built successfully!"
    else
        print_error "Failed to build the following images: ${failed_builds[*]}"
        return 1
    fi
}

# Function to build all images in parallel
build_parallel() {
    print_status "Building images in parallel..."
    
    # Build base image first (others depend on it)
    if ! build_image "Dockerfile" "latest"; then
        print_error "Failed to build base image. Cannot continue with parallel builds."
        return 1
    fi
    
    # Build other images in parallel
    local pids=()
    
    # Development image
    (build_image "Dockerfile.dev" "dev" && push_image "dev") &
    pids+=($!)
    
    # Jupyter image  
    (build_image "Dockerfile.jupyter" "jupyter" && push_image "jupyter") &
    pids+=($!)
    
    # GPU image (if available)
    if check_nvidia_docker; then
        (build_image "Dockerfile.gpu" "gpu" && push_image "gpu") &
        pids+=($!)
    fi
    
    # Wait for all background jobs
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            failed=1
        fi
    done
    
    if [ $failed -eq 0 ]; then
        print_success "All parallel builds completed successfully!"
    else
        print_error "Some parallel builds failed"
        return 1
    fi
}

# Function to create multi-architecture images
build_multiarch() {
    print_status "Building multi-architecture images..."
    
    # Create builder if it doesn't exist
    docker buildx create --name quantrs2-builder --use --bootstrap || true
    
    # Build for multiple architectures
    local platforms="linux/amd64,linux/arm64"
    
    # Base image
    docker buildx build \
        --platform "${platforms}" \
        -f "docker/Dockerfile" \
        -t "${REGISTRY}:latest" \
        -t "${REGISTRY}:${VERSION}" \
        ${BUILD_ARGS} \
        --push \
        ..
    
    # Development image
    docker buildx build \
        --platform "${platforms}" \
        -f "docker/Dockerfile.dev" \
        -t "${REGISTRY}:dev" \
        -t "${REGISTRY}:${VERSION}-dev" \
        ${BUILD_ARGS} \
        --push \
        ..
    
    print_success "Multi-architecture builds completed!"
}

# Function to test built images
test_images() {
    print_status "Testing built images..."
    
    local test_failed=false
    
    # Test base image
    if docker run --rm "${REGISTRY}:latest" python -c "import quantrs2; print('Base image OK')"; then
        print_success "Base image test passed"
    else
        print_error "Base image test failed"
        test_failed=true
    fi
    
    # Test development image
    if docker run --rm "${REGISTRY}:dev" python -c "import quantrs2; import pytest; print('Dev image OK')"; then
        print_success "Development image test passed"
    else
        print_error "Development image test failed"
        test_failed=true
    fi
    
    # Test Jupyter image
    if docker run --rm "${REGISTRY}:jupyter" python -c "import quantrs2; import jupyter; print('Jupyter image OK')"; then
        print_success "Jupyter image test passed"
    else
        print_error "Jupyter image test failed"
        test_failed=true
    fi
    
    if [ "$test_failed" = true ]; then
        print_error "Some image tests failed"
        return 1
    else
        print_success "All image tests passed!"
    fi
}

# Function to clean up old images
cleanup() {
    print_status "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old versions (keep last 3)
    docker images "${REGISTRY}" --format "table {{.Tag}}\t{{.CreatedAt}}" | \
    tail -n +2 | sort -k2 -r | tail -n +4 | awk '{print $1}' | \
    while read tag; do
        if [[ "$tag" != "latest" && "$tag" != "dev" && "$tag" != "jupyter" && "$tag" != "gpu" ]]; then
            print_status "Removing old image: ${REGISTRY}:${tag}"
            docker rmi "${REGISTRY}:${tag}" || true
        fi
    done
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    cat << EOF
QuantRS2 Docker Build Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    build       Build all Docker images (default)
    test        Test built images
    push        Push images to registry
    clean       Clean up old images
    multiarch   Build multi-architecture images

Options:
    -r, --registry REGISTRY     Docker registry/namespace (default: quantrs2)
    -v, --version VERSION       Version tag (default: latest)
    -p, --push                  Push images after building
    -j, --parallel              Build images in parallel
    -m, --multiarch             Build multi-architecture images
    -t, --test                  Run tests after building
    -c, --clean                 Clean up after building
    -h, --help                  Show this help message

Environment Variables:
    DOCKER_REGISTRY            Docker registry/namespace
    VERSION                    Version tag
    BUILD_ARGS                 Additional docker build arguments
    PUSH                       Push images (true/false)
    PARALLEL                   Build in parallel (true/false)

Examples:
    $0                         Build all images
    $0 -p                      Build and push images
    $0 -j -t                   Build in parallel and test
    $0 --registry myregistry   Build with custom registry
    $0 clean                   Clean up old images

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -j|--parallel)
            PARALLEL=true
            shift
            ;;
        -m|--multiarch)
            MULTIARCH=true
            shift
            ;;
        -t|--test)
            TEST=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        build|test|push|clean|multiarch)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default command
COMMAND=${COMMAND:-"build"}

# Main execution
main() {
    print_status "QuantRS2 Docker Build Script"
    print_status "Registry: ${REGISTRY}"
    print_status "Version: ${VERSION}"
    print_status "Command: ${COMMAND}"
    
    # Check prerequisites
    check_docker
    
    case $COMMAND in
        build)
            if [ "$MULTIARCH" = true ]; then
                build_multiarch
            elif [ "$PARALLEL" = true ]; then
                build_parallel
            else
                build_sequential
            fi
            
            if [ "$TEST" = true ]; then
                test_images
            fi
            
            if [ "$CLEAN" = true ]; then
                cleanup
            fi
            ;;
        test)
            test_images
            ;;
        push)
            push_image "latest"
            push_image "dev"
            push_image "jupyter"
            if check_nvidia_docker; then
                push_image "gpu"
            fi
            ;;
        clean)
            cleanup
            ;;
        multiarch)
            build_multiarch
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi