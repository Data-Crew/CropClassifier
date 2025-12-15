#!/bin/bash

# Helper script for CropClassifier Docker environment

set -e

echo "🐳 CropClassifier Docker Helper"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if nvidia-docker is available
check_nvidia_docker() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ nvidia-smi not available. Verify NVIDIA drivers are installed.${NC}"
        exit 1
    fi
    
    if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠️  nvidia-container-toolkit is not configured correctly.${NC}"
        echo "Installing nvidia-container-toolkit..."
        
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        
        echo -e "${GREEN}✅ nvidia-container-toolkit installed successfully.${NC}"
    else
        echo -e "${GREEN}✅ GPU support verified.${NC}"
    fi
}

# Build the Docker image
build_image() {
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    docker compose build
    echo -e "${GREEN}✅ Image built successfully.${NC}"
}

# Start main container (for bash scripts)
start_main() {
    echo -e "${YELLOW}🚀 Starting main container (bash mode)...${NC}"
    docker compose up -d cropclassifier
    echo -e "${GREEN}✅ Container started.${NC}"
    echo ""
    echo -e "${GREEN}To execute bash scripts:${NC}"
    echo "  docker compose exec cropclassifier bash"
    echo "  # then run: bash build_training_data.sh multiple all"
}

# Start JupyterLab container
start_jupyter() {
    echo -e "${YELLOW}🚀 Starting JupyterLab container...${NC}"
    docker compose --profile jupyter up -d jupyter
    echo -e "${GREEN}✅ JupyterLab started.${NC}"
    echo ""
    echo -e "${GREEN}📔 JupyterLab available at: http://localhost:8888${NC}"
    echo ""
    echo "To view logs:"
    echo "  docker compose logs -f jupyter"
}

# Start both containers
start_all() {
    echo -e "${YELLOW}🚀 Starting all containers...${NC}"
    docker compose --profile jupyter up -d
    echo -e "${GREEN}✅ All containers started.${NC}"
    echo ""
    echo -e "${GREEN}📔 JupyterLab available at: http://localhost:8888${NC}"
    echo -e "${GREEN}💻 Main container running in bash mode${NC}"
}

# Stop containers
stop_container() {
    echo -e "${YELLOW}🛑 Stopping containers...${NC}"
    docker compose --profile jupyter down
    echo -e "${GREEN}✅ Containers stopped.${NC}"
}

# Execute shell in main container
shell() {
    echo -e "${YELLOW}🐚 Opening shell in main container...${NC}"
    docker compose exec cropclassifier /bin/bash
}

# Execute a bash script in the container
exec_script() {
    if [ -z "$2" ]; then
        echo -e "${RED}❌ Please provide a script to execute${NC}"
        echo "Example: ./docker-run.sh exec 'bash build_training_data.sh multiple all'"
        exit 1
    fi
    
    echo -e "${YELLOW}⚙️  Executing: $2${NC}"
    docker compose exec cropclassifier bash -c "$2"
}

# Show container logs
logs() {
    if [ "$2" == "jupyter" ]; then
        docker compose logs -f jupyter
    else
        docker compose logs -f cropclassifier
    fi
}

# Show help
show_help() {
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  check        - Verify GPU support"
    echo "  build        - Build Docker image"
    echo "  start        - Start main container (bash mode)"
    echo "  jupyter      - Start JupyterLab container"
    echo "  all          - Start both containers"
    echo "  stop         - Stop all containers"
    echo "  restart      - Restart all containers"
    echo "  shell        - Open shell in main container"
    echo "  exec <cmd>   - Execute command in container"
    echo "  logs [srv]   - Show logs (optional: jupyter)"
    echo "  help         - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh build"
    echo "  ./docker-run.sh start"
    echo "  ./docker-run.sh jupyter"
    echo "  ./docker-run.sh shell"
    echo "  ./docker-run.sh exec 'bash build_training_data.sh multiple 2'"
    echo "  ./docker-run.sh logs jupyter"
}

# Main
case "$1" in
    check)
        check_nvidia_docker
        ;;
    build)
        check_nvidia_docker
        build_image
        ;;
    start)
        check_nvidia_docker
        start_main
        ;;
    jupyter)
        check_nvidia_docker
        start_jupyter
        ;;
    all)
        check_nvidia_docker
        start_all
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        sleep 2
        start_all
        ;;
    shell)
        shell
        ;;
    exec)
        exec_script "$@"
        ;;
    logs)
        logs "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            show_help
        else
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo ""
            show_help
            exit 1
        fi
        ;;
esac
