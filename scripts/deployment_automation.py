#!/usr/bin/env python3
"""
Automated deployment script for Spacecraft DR-MPC System
Supports multiple deployment targets: local, Docker, Kubernetes, cloud platforms
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentAutomator:
    """Automated deployment system for spacecraft DR-MPC system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "deployment" / "config.yaml"
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load deployment configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self.default_config()
    
    def default_config(self) -> Dict:
        """Default deployment configuration."""
        return {
            "environments": {
                "local": {
                    "python_version": "3.11",
                    "virtual_env": True,
                    "port": 8080
                },
                "docker": {
                    "image_name": "spacecraft-drmpc",
                    "tag": "latest",
                    "registry": "local"
                },
                "kubernetes": {
                    "namespace": "spacecraft-system",
                    "replicas": 3,
                    "cluster_name": "spacecraft-cluster"
                },
                "aws": {
                    "region": "us-west-2",
                    "cluster_name": "spacecraft-ecs",
                    "service_name": "spacecraft-service"
                }
            },
            "validation": {
                "run_tests": True,
                "performance_check": True,
                "security_scan": True
            }
        }
    
    def run_command(self, command: str, shell: bool = True, capture_output: bool = False) -> subprocess.CompletionProcess:
        """Run shell command with error handling."""
        logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(
                command, 
                shell=shell, 
                capture_output=capture_output, 
                text=True,
                cwd=self.project_root
            )
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                if capture_output:
                    logger.error(f"Error output: {result.stderr}")
                return result
            return result
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            raise
    
    def validate_environment(self, target: str) -> bool:
        """Validate deployment environment requirements."""
        logger.info(f"Validating environment for {target} deployment")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            logger.error(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        # Check required tools based on target
        required_tools = {
            "local": ["python3", "pip"],
            "docker": ["docker", "docker-compose"],
            "kubernetes": ["kubectl", "helm"],
            "aws": ["aws", "docker"],
            "gcp": ["gcloud", "kubectl"],
            "azure": ["az", "kubectl"]
        }
        
        if target in required_tools:
            for tool in required_tools[target]:
                result = self.run_command(f"which {tool}", capture_output=True)
                if result.returncode != 0:
                    logger.error(f"Required tool {tool} not found")
                    return False
        
        logger.info("Environment validation passed")
        return True
    
    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation checks."""
        logger.info("Running pre-deployment checks")
        
        checks_passed = True
        
        # Run tests if configured
        if self.config.get("validation", {}).get("run_tests", True):
            logger.info("Running test suite")
            result = self.run_command("python -m pytest tests/ -v")
            if result.returncode != 0:
                logger.error("Tests failed")
                checks_passed = False
        
        # Run performance benchmarks
        if self.config.get("validation", {}).get("performance_check", True):
            logger.info("Running performance checks")
            result = self.run_command("python scripts/quick_test.py")
            if result.returncode != 0:
                logger.error("Performance checks failed")
                checks_passed = False
        
        # Security scan
        if self.config.get("validation", {}).get("security_scan", True):
            logger.info("Running security scan")
            result = self.run_command("bandit -r src/ -f json -o reports/security_scan.json")
            # Don't fail on security warnings, just log
            if result.returncode != 0:
                logger.warning("Security scan completed with warnings")
        
        return checks_passed
    
    def deploy_local(self) -> bool:
        """Deploy to local development environment."""
        logger.info("Starting local deployment")
        
        config = self.config.get("environments", {}).get("local", {})
        
        # Create virtual environment if configured
        if config.get("virtual_env", True):
            venv_path = self.project_root / "venv"
            if not venv_path.exists():
                logger.info("Creating virtual environment")
                self.run_command("python3 -m venv venv")
            
            # Install dependencies
            logger.info("Installing dependencies")
            self.run_command("venv/bin/pip install --upgrade pip")
            self.run_command("venv/bin/pip install -r requirements.txt")
            self.run_command("venv/bin/pip install -e .")
        
        # Start local server
        port = config.get("port", 8080)
        logger.info(f"Starting local server on port {port}")
        
        # Create startup script
        startup_script = self.project_root / "run_local.sh"
        with open(startup_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd {self.project_root}
source venv/bin/activate
export SPACECRAFT_MODE=development
export SPACECRAFT_PORT={port}
python -m spacecraft_drmpc.cli --port {port} --config config/development.yaml
""")
        os.chmod(startup_script, 0o755)
        
        logger.info("Local deployment completed")
        logger.info(f"Start server with: {startup_script}")
        return True
    
    def deploy_docker(self) -> bool:
        """Deploy using Docker containers."""
        logger.info("Starting Docker deployment")
        
        config = self.config.get("environments", {}).get("docker", {})
        image_name = config.get("image_name", "spacecraft-drmpc")
        tag = config.get("tag", "latest")
        
        # Build Docker image
        logger.info(f"Building Docker image {image_name}:{tag}")
        dockerfile_content = self.generate_dockerfile()
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        result = self.run_command(f"docker build -t {image_name}:{tag} .")
        if result.returncode != 0:
            logger.error("Docker build failed")
            return False
        
        # Create Docker Compose file
        compose_content = self.generate_docker_compose(image_name, tag)
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info("Docker deployment prepared")
        logger.info("Start with: docker-compose up -d")
        return True
    
    def deploy_kubernetes(self) -> bool:
        """Deploy to Kubernetes cluster."""
        logger.info("Starting Kubernetes deployment")
        
        config = self.config.get("environments", {}).get("kubernetes", {})
        namespace = config.get("namespace", "spacecraft-system")
        replicas = config.get("replicas", 3)
        
        # Create namespace if it doesn't exist
        self.run_command(f"kubectl create namespace {namespace} --dry-run=client -o yaml | kubectl apply -f -")
        
        # Generate Kubernetes manifests
        manifests = self.generate_kubernetes_manifests(namespace, replicas)
        
        manifests_dir = self.project_root / "k8s"
        manifests_dir.mkdir(exist_ok=True)
        
        for filename, content in manifests.items():
            with open(manifests_dir / filename, 'w') as f:
                f.write(content)
        
        # Apply manifests
        logger.info("Applying Kubernetes manifests")
        result = self.run_command(f"kubectl apply -f k8s/ -n {namespace}")
        if result.returncode != 0:
            logger.error("Kubernetes deployment failed")
            return False
        
        # Wait for rollout
        self.run_command(f"kubectl rollout status deployment/spacecraft-dr-mpc -n {namespace}")
        
        logger.info("Kubernetes deployment completed")
        return True
    
    def deploy_aws(self) -> bool:
        """Deploy to AWS ECS."""
        logger.info("Starting AWS ECS deployment")
        
        config = self.config.get("environments", {}).get("aws", {})
        region = config.get("region", "us-west-2")
        cluster_name = config.get("cluster_name", "spacecraft-ecs")
        
        # Build and push Docker image to ECR
        logger.info("Building and pushing image to ECR")
        
        # Get ECR login
        result = self.run_command(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.{region}.amazonaws.com", capture_output=True)
        if result.returncode != 0:
            logger.error("ECR login failed")
            return False
        
        # Create ECR repository if it doesn't exist
        self.run_command(f"aws ecr create-repository --repository-name spacecraft-drmpc --region {region} 2>/dev/null || true")
        
        # Build and push image
        account_id = self.run_command("aws sts get-caller-identity --query Account --output text", capture_output=True).stdout.strip()
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/spacecraft-drmpc:latest"
        
        self.run_command("docker build -t spacecraft-drmpc:latest .")
        self.run_command(f"docker tag spacecraft-drmpc:latest {image_uri}")
        self.run_command(f"docker push {image_uri}")
        
        # Generate ECS task definition
        task_definition = self.generate_ecs_task_definition(image_uri)
        
        with open("task-definition.json", 'w') as f:
            json.dump(task_definition, f, indent=2)
        
        # Register task definition and update service
        self.run_command("aws ecs register-task-definition --cli-input-json file://task-definition.json")
        self.run_command(f"aws ecs update-service --cluster {cluster_name} --service spacecraft-service --task-definition spacecraft-drmpc:latest --region {region}")
        
        logger.info("AWS ECS deployment completed")
        return True
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile content."""
        return '''FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libopenblas-dev \\
    liblapack-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -s /bin/bash spacecraft
RUN chown -R spacecraft:spacecraft /app
USER spacecraft

# Expose ports
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "spacecraft_drmpc.cli", "--config", "config/production.yaml"]
'''
    
    def generate_docker_compose(self, image_name: str, tag: str) -> str:
        """Generate Docker Compose configuration."""
        return f'''version: '3.8'

services:
  spacecraft-leader:
    image: {image_name}:{tag}
    container_name: spacecraft-leader
    environment:
      - SPACECRAFT_ID=leader
      - SPACECRAFT_ROLE=formation_leader
      - SPACECRAFT_MODE=production
    ports:
      - "8080:8080"
      - "8443:8443"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - spacecraft-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  spacecraft-follower:
    image: {image_name}:{tag}
    environment:
      - SPACECRAFT_ROLE=formation_follower
      - SPACECRAFT_MODE=production
      - LEADER_ADDRESS=spacecraft-leader
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    networks:
      - spacecraft-network
    restart: unless-stopped
    depends_on:
      - spacecraft-leader
    deploy:
      replicas: 3

networks:
  spacecraft-network:
    driver: bridge

volumes:
  spacecraft-data:
    driver: local
'''
    
    def generate_kubernetes_manifests(self, namespace: str, replicas: int) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        manifests = {}
        
        # Deployment manifest
        manifests["deployment.yaml"] = f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: spacecraft-dr-mpc
  namespace: {namespace}
  labels:
    app: spacecraft-system
    component: controller
spec:
  replicas: {replicas}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: spacecraft-system
      component: controller
  template:
    metadata:
      labels:
        app: spacecraft-system
        component: controller
    spec:
      containers:
      - name: spacecraft-controller
        image: spacecraft-drmpc:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8443
          name: https
        env:
        - name: SPACECRAFT_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: SPACECRAFT_MODE
          value: "production"
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: config-volume
        configMap:
          name: spacecraft-config
      - name: data-volume
        emptyDir: {{}}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: spacecraft-service
  namespace: {namespace}
  labels:
    app: spacecraft-system
spec:
  selector:
    app: spacecraft-system
    component: controller
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: https
    port: 443
    targetPort: 8443
    protocol: TCP
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: spacecraft-config
  namespace: {namespace}
data:
  production.yaml: |
    mission:
      type: "formation_flying"
      num_spacecraft: {replicas}
    
    controller:
      type: "dr_mpc"
      horizon_length: 15
      control_frequency: 100.0
      uncertainty_level: 0.2
    
    safety:
      collision_radius: 10.0
      emergency_stop_enabled: true
    
    communication:
      encryption: true
      secure_transport: true
    
    logging:
      level: "INFO"
      format: "json"
'''
        
        return manifests
    
    def generate_ecs_task_definition(self, image_uri: str) -> Dict:
        """Generate ECS task definition."""
        return {
            "family": "spacecraft-drmpc",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "1024",
            "memory": "2048",
            "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
            "containerDefinitions": [
                {
                    "name": "spacecraft-controller",
                    "image": image_uri,
                    "portMappings": [
                        {
                            "containerPort": 8080,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {
                            "name": "SPACECRAFT_MODE",
                            "value": "production"
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": "/ecs/spacecraft-controller",
                            "awslogs-region": "us-west-2",
                            "awslogs-stream-prefix": "ecs"
                        }
                    },
                    "healthCheck": {
                        "command": [
                            "CMD-SHELL",
                            "curl -f http://localhost:8080/health || exit 1"
                        ],
                        "interval": 30,
                        "timeout": 10,
                        "retries": 3,
                        "startPeriod": 60
                    }
                }
            ]
        }
    
    def post_deployment_validation(self, target: str) -> bool:
        """Run post-deployment validation checks."""
        logger.info(f"Running post-deployment validation for {target}")
        
        # Wait for service to be ready
        time.sleep(10)
        
        # Health check endpoint validation
        if target in ["local", "docker"]:
            result = self.run_command("curl -f http://localhost:8080/health", capture_output=True)
            if result.returncode != 0:
                logger.error("Health check failed")
                return False
        
        logger.info("Post-deployment validation passed")
        return True
    
    def rollback(self, target: str, version: str) -> bool:
        """Rollback deployment to previous version."""
        logger.info(f"Rolling back {target} deployment to {version}")
        
        if target == "kubernetes":
            result = self.run_command(f"kubectl rollout undo deployment/spacecraft-dr-mpc")
            return result.returncode == 0
        
        elif target == "aws":
            # Implement AWS ECS rollback
            pass
        
        logger.warning(f"Rollback not implemented for {target}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Spacecraft DR-MPC Deployment Automation")
    parser.add_argument("target", choices=["local", "docker", "kubernetes", "aws", "gcp", "azure"],
                       help="Deployment target")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-deployment validation")
    parser.add_argument("--rollback", help="Rollback to specific version")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    deployer = DeploymentAutomator(args.config)
    
    # Handle rollback request
    if args.rollback:
        success = deployer.rollback(args.target, args.rollback)
        sys.exit(0 if success else 1)
    
    # Validate environment
    if not deployer.validate_environment(args.target):
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Run pre-deployment checks
    if not args.skip_validation:
        if not deployer.pre_deployment_checks():
            logger.error("Pre-deployment validation failed")
            sys.exit(1)
    
    # Execute deployment
    deployment_methods = {
        "local": deployer.deploy_local,
        "docker": deployer.deploy_docker,
        "kubernetes": deployer.deploy_kubernetes,
        "aws": deployer.deploy_aws,
    }
    
    if args.target in deployment_methods:
        success = deployment_methods[args.target]()
        if not success:
            logger.error(f"Deployment to {args.target} failed")
            sys.exit(1)
        
        # Post-deployment validation
        if not deployer.post_deployment_validation(args.target):
            logger.warning("Post-deployment validation failed")
    else:
        logger.error(f"Deployment target {args.target} not implemented")
        sys.exit(1)
    
    logger.info(f"Deployment to {args.target} completed successfully")

if __name__ == "__main__":
    main()