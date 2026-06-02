#!/bin/bash
set -e

# ─────────────────────────────────────────────
#  PEAG — Private Enterprise AI Gateway
#  Installer v0.1.0
# ─────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_banner() {
  echo -e "${BLUE}"
  echo "  ██████╗ ███████╗ █████╗  ██████╗ "
  echo "  ██╔══██╗██╔════╝██╔══██╗██╔════╝ "
  echo "  ██████╔╝█████╗  ███████║██║  ███╗"
  echo "  ██╔═══╝ ██╔══╝  ██╔══██║██║   ██║"
  echo "  ██║     ███████╗██║  ██║╚██████╔╝"
  echo "  ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ "
  echo -e "${NC}"
  echo "  Private Enterprise AI Gateway"
  echo "  Version 0.1.0"
  echo ""
  echo "  Your private AI — no cloud, no subscriptions, no data leaving your building."
  echo ""
}

check_requirements() {
  echo -e "${BLUE}Checking system requirements...${NC}"

  if [ ! -f /etc/os-release ]; then
    echo -e "${RED}Error: Unsupported OS. PEAG requires Ubuntu or Debian.${NC}"
    exit 1
  fi

  TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
  if [ "$TOTAL_RAM" -lt 8 ]; then
    echo -e "${YELLOW}Warning: Less than 8GB RAM. Only small models will run reliably.${NC}"
  else
    echo -e "${GREEN}RAM: ${TOTAL_RAM}GB detected.${NC}"
  fi

  FREE_DISK=$(df -BG / | awk 'NR==2{print $4}' | tr -d 'G')
  if [ "$FREE_DISK" -lt 20 ]; then
    echo -e "${RED}Error: Less than 20GB free disk space. PEAG needs at least 20GB.${NC}"
    exit 1
  else
    echo -e "${GREEN}Disk: ${FREE_DISK}GB free.${NC}"
  fi

  if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}GPU: $GPU_NAME ($GPU_VRAM VRAM)${NC}"
    HAS_GPU=true
  else
    echo -e "${YELLOW}No GPU detected. Running CPU-only mode (slower inference).${NC}"
    HAS_GPU=false
  fi

  echo -e "${GREEN}Requirements check passed.${NC}"
  echo ""
}

install_dependencies() {
  echo -e "${BLUE}Installing dependencies...${NC}"
  apt-get update -qq
  apt-get install -y -qq curl wget git jq

  if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker $USER
  fi

  echo -e "${GREEN}Dependencies ready.${NC}"
  echo ""
}

install_k3s() {
  echo -e "${BLUE}Installing Kubernetes (k3s)...${NC}"

  curl -sfL https://get.k3s.io | sh -s - \
    --disable traefik \
    --disable servicelb \
    --write-kubeconfig-mode 644

  sleep 10
  until kubectl get nodes 2>/dev/null | grep -q "Ready"; do
    echo "Waiting for cluster..."
    sleep 3
  done

  mkdir -p ~/.kube
  cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
  chmod 600 ~/.kube/config

  echo -e "${GREEN}Kubernetes ready.${NC}"
  echo ""
}

select_models() {
  echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  Select your AI models${NC}"
  echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "${CYAN}  FAST MODELS — Run on CPU, quick responses, low RAM${NC}"
  echo ""
  echo "  [1]  TinyLlama 1B      637MB   Basic Q&A, extremely fast"
  echo "  [2]  Phi-3 Mini 3.8B   2.2GB   Smart and efficient  ★ Recommended"
  echo "  [3]  Gemma2 2B         1.6GB   Google's compact model, great quality"
  echo "  [4]  Qwen2.5 1.5B      1.0GB   Excellent multilingual support"
  echo "  [5]  SmolLM2 1.7B      1.0GB   Tiny but surprisingly capable"
  echo ""
  echo -e "${CYAN}  SMART MODELS — GPU recommended, high quality responses${NC}"
  echo ""
  echo "  [6]  Llama3.2 8B       4.7GB   Meta's best open model  ★ Recommended"
  echo "  [7]  Mistral 7B        4.1GB   Excellent reasoning and code"
  echo "  [8]  Gemma2 9B         5.4GB   Google's powerful model"
  echo "  [9]  Qwen2.5 7B        4.4GB   Great for multilingual and code"
  echo "  [10] DeepSeek-R1 7B    4.7GB   Strong reasoning model"
  echo "  [11] Phi-4 Mini 14B    8.5GB   Microsoft's frontier small model"
  echo ""
  echo -e "${CYAN}  CODE MODELS — Optimised for software development${NC}"
  echo ""
  echo "  [12] CodeLlama 7B      3.8GB   Meta's code specialist"
  echo "  [13] DeepSeek-Coder 6.7B 3.8GB Best open source code model"
  echo "  [14] Qwen2.5-Coder 7B  4.4GB   Excellent code generation"
  echo "  [15] StarCoder2 7B     4.0GB   Multi-language code model"
  echo ""
  echo -e "${CYAN}  LARGE MODELS — High end GPU required (16GB+ VRAM)${NC}"
  echo ""
  echo "  [16] Llama3.1 70B      40GB    Near GPT-4 quality"
  echo "  [17] Mixtral 8x7B      26GB    Fast mixture-of-experts model"
  echo "  [18] Qwen2.5 72B       41GB    Frontier open source model"
  echo "  [19] DeepSeek-R1 70B   40GB    Top reasoning model"
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo "  Enter model numbers separated by spaces (e.g. 2 6 13):"
  echo "  Or press Enter to install the recommended defaults (Phi-3 + Llama3.2):"
  echo ""
  read -r MODEL_SELECTION

  # Default to recommended if nothing selected
  if [ -z "$MODEL_SELECTION" ]; then
    MODEL_SELECTION="2 6"
    echo -e "${YELLOW}Installing recommended defaults: Phi-3 + Llama3.2${NC}"
  fi

  SELECTED_MODELS=()
  SELECTED_NAMES=()

  for num in $MODEL_SELECTION; do
    case $num in
      1)  SELECTED_MODELS+=("tinyllama");          SELECTED_NAMES+=("TinyLlama 1B") ;;
      2)  SELECTED_MODELS+=("phi3");               SELECTED_NAMES+=("Phi-3 Mini") ;;
      3)  SELECTED_MODELS+=("gemma2:2b");          SELECTED_NAMES+=("Gemma2 2B") ;;
      4)  SELECTED_MODELS+=("qwen2.5:1.5b");       SELECTED_NAMES+=("Qwen2.5 1.5B") ;;
      5)  SELECTED_MODELS+=("smollm2");            SELECTED_NAMES+=("SmolLM2 1.7B") ;;
      6)  SELECTED_MODELS+=("llama3.2");           SELECTED_NAMES+=("Llama3.2 8B") ;;
      7)  SELECTED_MODELS+=("mistral");            SELECTED_NAMES+=("Mistral 7B") ;;
      8)  SELECTED_MODELS+=("gemma2:9b");          SELECTED_NAMES+=("Gemma2 9B") ;;
      9)  SELECTED_MODELS+=("qwen2.5:7b");         SELECTED_NAMES+=("Qwen2.5 7B") ;;
      10) SELECTED_MODELS+=("deepseek-r1:7b");     SELECTED_NAMES+=("DeepSeek-R1 7B") ;;
      11) SELECTED_MODELS+=("phi4-mini");          SELECTED_NAMES+=("Phi-4 Mini 14B") ;;
      12) SELECTED_MODELS+=("codellama:7b");       SELECTED_NAMES+=("CodeLlama 7B") ;;
      13) SELECTED_MODELS+=("deepseek-coder:6.7b"); SELECTED_NAMES+=("DeepSeek-Coder 6.7B") ;;
      14) SELECTED_MODELS+=("qwen2.5-coder:7b");  SELECTED_NAMES+=("Qwen2.5-Coder 7B") ;;
      15) SELECTED_MODELS+=("starcoder2:7b");      SELECTED_NAMES+=("StarCoder2 7B") ;;
      16) SELECTED_MODELS+=("llama3.1:70b");       SELECTED_NAMES+=("Llama3.1 70B") ;;
      17) SELECTED_MODELS+=("mixtral:8x7b");       SELECTED_NAMES+=("Mixtral 8x7B") ;;
      18) SELECTED_MODELS+=("qwen2.5:72b");        SELECTED_NAMES+=("Qwen2.5 72B") ;;
      19) SELECTED_MODELS+=("deepseek-r1:70b");    SELECTED_NAMES+=("DeepSeek-R1 70B") ;;
      *)  echo -e "${YELLOW}Unknown option $num, skipping.${NC}" ;;
    esac
  done

  if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    echo "No valid models selected. Installing Phi-3 as default."
    SELECTED_MODELS=("phi3")
    SELECTED_NAMES=("Phi-3 Mini")
  fi

  echo ""
  echo -e "${GREEN}You selected:${NC}"
  for name in "${SELECTED_NAMES[@]}"; do
    echo -e "  ${GREEN}✓${NC} $name"
  done
  echo ""

  # Calculate approximate disk usage
  TOTAL_SIZE=0
  for model in "${SELECTED_MODELS[@]}"; do
    case $model in
      tinyllama)    TOTAL_SIZE=$((TOTAL_SIZE + 1)) ;;
      phi3)         TOTAL_SIZE=$((TOTAL_SIZE + 3)) ;;
      gemma2:2b)    TOTAL_SIZE=$((TOTAL_SIZE + 2)) ;;
      qwen2.5:1.5b) TOTAL_SIZE=$((TOTAL_SIZE + 1)) ;;
      smollm2)      TOTAL_SIZE=$((TOTAL_SIZE + 1)) ;;
      llama3.2)     TOTAL_SIZE=$((TOTAL_SIZE + 5)) ;;
      mistral)      TOTAL_SIZE=$((TOTAL_SIZE + 5)) ;;
      gemma2:9b)    TOTAL_SIZE=$((TOTAL_SIZE + 6)) ;;
      qwen2.5:7b)   TOTAL_SIZE=$((TOTAL_SIZE + 5)) ;;
      deepseek-r1:7b) TOTAL_SIZE=$((TOTAL_SIZE + 5)) ;;
      phi4-mini)    TOTAL_SIZE=$((TOTAL_SIZE + 9)) ;;
      codellama:7b) TOTAL_SIZE=$((TOTAL_SIZE + 4)) ;;
      deepseek-coder:6.7b) TOTAL_SIZE=$((TOTAL_SIZE + 4)) ;;
      qwen2.5-coder:7b) TOTAL_SIZE=$((TOTAL_SIZE + 5)) ;;
      starcoder2:7b) TOTAL_SIZE=$((TOTAL_SIZE + 4)) ;;
      llama3.1:70b) TOTAL_SIZE=$((TOTAL_SIZE + 41)) ;;
      mixtral:8x7b) TOTAL_SIZE=$((TOTAL_SIZE + 27)) ;;
      qwen2.5:72b)  TOTAL_SIZE=$((TOTAL_SIZE + 42)) ;;
      deepseek-r1:70b) TOTAL_SIZE=$((TOTAL_SIZE + 41)) ;;
    esac
  done

  echo -e "${YELLOW}Approximate download size: ~${TOTAL_SIZE}GB${NC}"
  echo ""
  echo "Proceed with installation? (y/n)"
  read -r CONFIRM
  if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Installation cancelled."
    exit 0
  fi
}

install_peag() {
  echo -e "${BLUE}Installing PEAG...${NC}"

  PEAG_RAW="https://raw.githubusercontent.com/ade-daemon/peag/main"

  kubectl apply -f $PEAG_RAW/k8s/namespaces/ai-namespace.yaml
  kubectl apply -f $PEAG_RAW/k8s/crds/modelconfig-crd.yaml
  kubectl apply -f $PEAG_RAW/k8s/redis/redis-deployment.yaml
  kubectl apply -f $PEAG_RAW/k8s/redis/redis-service.yaml
  kubectl apply -f $PEAG_RAW/k8s/gateway/gateway-rbac.yaml
  kubectl apply -f $PEAG_RAW/k8s/gateway/gateway-deployment.yaml
  kubectl apply -f $PEAG_RAW/k8s/gateway/gateway-service.yaml
  kubectl apply -f $PEAG_RAW/k8s/open-webui/open-webui-deployment.yaml
  kubectl apply -f $PEAG_RAW/k8s/open-webui/open-webui-service.yaml

  echo "Waiting for PEAG to be ready..."
  kubectl wait --for=condition=ready pod -l app=peag-gateway -n ai --timeout=120s
  kubectl wait --for=condition=ready pod -l app=redis -n ai --timeout=60s

  echo -e "${GREEN}PEAG installed.${NC}"
  echo ""
}

pull_models() {
  echo -e "${BLUE}Pulling selected models (this may take a while)...${NC}"
  echo ""

  GATEWAY_POD=$(kubectl get pod -n ai -l app=peag-gateway -o jsonpath='{.items[0].metadata.name}')

  for i in "${!SELECTED_MODELS[@]}"; do
    model="${SELECTED_MODELS[$i]}"
    name="${SELECTED_NAMES[$i]}"
    echo -e "${BLUE}Pulling $name ($model)...${NC}"
    kubectl exec -n ai $GATEWAY_POD -- ollama pull $model || \
      echo -e "${YELLOW}Warning: Could not pull $model. You can pull it later with: peag pull $model${NC}"
    echo -e "${GREEN}✓ $name ready${NC}"
    echo ""
  done
}

install_cli() {
  echo -e "${BLUE}Installing PEAG CLI...${NC}"

  cat > /usr/local/bin/peag << 'CLISCRIPT'
#!/bin/bash
GATEWAY_POD=$(kubectl get pod -n ai -l app=peag-gateway -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

case "$1" in
  pull)
    if [ -z "$2" ]; then
      echo "Usage: peag pull <model>"
      echo "Example: peag pull mistral"
      exit 1
    fi
    echo "Pulling $2..."
    kubectl exec -n ai $GATEWAY_POD -- ollama pull $2
    echo "Done. $2 is ready."
    ;;
  list)
    echo "Available models:"
    kubectl exec -n ai $GATEWAY_POD -- ollama list
    ;;
  status)
    echo "PEAG Status:"
    echo ""
    kubectl get pods -n ai
    ;;
  stop)
    docker stop peag-control-plane peag-worker peag-worker2 2>/dev/null || \
    systemctl stop k3s
    echo "PEAG stopped."
    ;;
  start)
    docker start peag-control-plane peag-worker peag-worker2 2>/dev/null || \
    systemctl start k3s
    echo "PEAG started."
    ;;
  *)
    echo "PEAG — Private Enterprise AI Gateway"
    echo ""
    echo "Commands:"
    echo "  peag pull <model>   Pull a new AI model"
    echo "  peag list           List installed models"
    echo "  peag status         Show running services"
    echo "  peag start          Start PEAG"
    echo "  peag stop           Stop PEAG"
    echo ""
    echo "Available models: tinyllama, phi3, llama3.2, mistral, gemma2, codellama"
    echo "Full list: github.com/ade-daemon/peag"
    ;;
esac
CLISCRIPT

  chmod +x /usr/local/bin/peag
  echo -e "${GREEN}PEAG CLI installed. Type 'peag' to get started.${NC}"
  echo ""
}

print_success() {
  LOCAL_IP=$(hostname -I | awk '{print $1}')

  echo ""
  echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}${BOLD}  PEAG is ready!${NC}"
  echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "  ${BOLD}Open WebUI:${NC}    http://$LOCAL_IP:3000"
  echo -e "  ${BOLD}PEAG Gateway:${NC}  http://$LOCAL_IP:8000"
  echo ""
  echo -e "  ${BOLD}Installed models:${NC}"
  for name in "${SELECTED_NAMES[@]}"; do
    echo -e "    ${GREEN}✓${NC} $name"
  done
  echo ""
  echo -e "  ${BOLD}Share this URL with your team:${NC}"
  echo -e "  ${CYAN}http://$LOCAL_IP:3000${NC}"
  echo ""
  echo -e "  ${BOLD}Pull more models anytime:${NC}"
  echo -e "  ${CYAN}peag pull mistral${NC}"
  echo -e "  ${CYAN}peag pull codellama${NC}"
  echo -e "  ${CYAN}peag list${NC}"
  echo ""
  echo -e "${BLUE}  Your data stays on this machine. Always.${NC}"
  echo ""
}

# ── Run installer ─────────────────────────────────────────────────────────────
print_banner
check_requirements
install_dependencies
install_k3s
select_models
install_peag
pull_models
install_cli
print_success
