# Production Deployment Guide

This guide covers deploying Saplings applications to production environments with Docker, Kubernetes, and cloud platforms, including scaling, monitoring, and operational best practices.

## Table of Contents

- [Deployment Architecture](#deployment-architecture)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Environment Configuration](#environment-configuration)
- [Scaling Strategies](#scaling-strategies)
- [Health Checks & Monitoring](#health-checks--monitoring)
- [Load Balancing](#load-balancing)
- [Performance Optimization](#performance-optimization)
- [Security in Production](#security-in-production)
- [Operational Best Practices](#operational-best-practices)

## Deployment Architecture

### Recommended Production Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer]
    end
    
    subgraph "Application Tier"
        A1[Agent Instance 1]
        A2[Agent Instance 2]
        A3[Agent Instance N]
    end
    
    subgraph "Data Tier"
        DB[(Vector Database)]
        MEM[(Memory Store)]
        CACHE[(Redis Cache)]
    end
    
    subgraph "External Services"
        LLM[LLM APIs]
        MON[Monitoring]
        LOG[Logging]
    end
    
    LB --> A1
    LB --> A2
    LB --> A3
    
    A1 --> DB
    A1 --> MEM
    A1 --> CACHE
    A1 --> LLM
    
    A2 --> DB
    A2 --> MEM
    A2 --> CACHE
    A2 --> LLM
    
    A3 --> DB
    A3 --> MEM
    A3 --> CACHE
    A3 --> LLM
    
    A1 --> MON
    A2 --> MON
    A3 --> MON
    
    A1 --> LOG
    A2 --> LOG
    A3 --> LOG
```

### Components Overview

| Component | Purpose | Scaling | Persistence |
|-----------|---------|---------|------------|
| **Agent Instances** | Core application logic | Horizontal | Stateless |
| **Vector Database** | Document embeddings | Vertical/Horizontal | Persistent |
| **Memory Store** | Agent memory | Horizontal | Persistent |
| **Cache Layer** | Performance optimization | Horizontal | Volatile |
| **Load Balancer** | Traffic distribution | Active/Passive | Configuration |

## Docker Deployment

### Production Dockerfile

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r saplings && useradd -r -g saplings saplings

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=saplings:saplings . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R saplings:saplings /app

# Switch to non-root user
USER saplings

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "
import requests
import sys
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    sys.exit(0 if response.status_code == 200 else 1)
except:
    sys.exit(1)
"

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "main:app"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  saplings-app:
    build: .
    container_name: saplings-app
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/saplings
      - ENVIRONMENT=development
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    networks:
      - saplings-network

  redis:
    image: redis:7-alpine
    container_name: saplings-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - saplings-network

  postgres:
    image: postgres:15-alpine
    container_name: saplings-postgres
    environment:
      - POSTGRES_DB=saplings
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - saplings-network

  nginx:
    image: nginx:alpine
    container_name: saplings-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - saplings-app
    networks:
      - saplings-network

volumes:
  redis_data:
  postgres_data:

networks:
  saplings-network:
    driver: bridge
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  saplings-app:
    build:
      context: .
      target: production
    deploy:
      replicas: 3
      restart_policy:
        condition: any
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=${DATABASE_URL}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    secrets:
      - api_keys
      - db_password
    networks:
      - saplings-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 512M
    volumes:
      - redis_data:/data
    networks:
      - saplings-network

secrets:
  api_keys:
    file: ./secrets/api_keys.txt
  db_password:
    file: ./secrets/db_password.txt

volumes:
  redis_data:

networks:
  saplings-network:
    external: true
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: saplings
  labels:
    name: saplings

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: saplings-config
  namespace: saplings
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379"
  GASA_ENABLED: "true"
  MONITORING_ENABLED: "true"
  MAX_WORKERS: "4"
  TIMEOUT: "120"
```

### Secrets Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: saplings-secrets
  namespace: saplings
type: Opaque
data:
  # Base64 encoded values
  OPENAI_API_KEY: <base64-encoded-key>
  ANTHROPIC_API_KEY: <base64-encoded-key>
  DATABASE_URL: <base64-encoded-url>
  ENCRYPTION_KEY: <base64-encoded-key>

---
# Create secrets from command line:
# kubectl create secret generic saplings-secrets \
#   --from-literal=OPENAI_API_KEY=your-openai-key \
#   --from-literal=ANTHROPIC_API_KEY=your-anthropic-key \
#   --namespace=saplings
```

### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saplings-app
  namespace: saplings
  labels:
    app: saplings
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: saplings
  template:
    metadata:
      labels:
        app: saplings
    spec:
      containers:
      - name: saplings
        image: saplings:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: PORT
          value: "8000"
        envFrom:
        - configMapRef:
            name: saplings-config
        - secretRef:
            name: saplings-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: saplings-data-pvc
      - name: logs-volume
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: saplings-service
  namespace: saplings
  labels:
    app: saplings
spec:
  selector:
    app: saplings
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: saplings-ingress
  namespace: saplings
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: saplings-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: saplings-service
            port:
              number: 80
```

### Persistent Volume Claims

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: saplings-data-pvc
  namespace: saplings
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
# For Redis persistence
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-data-pvc
  namespace: saplings
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: standard
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: saplings-hpa
  namespace: saplings
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: saplings-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Cloud Platform Deployment

### AWS ECS Deployment

```yaml
# ecs-task-definition.json
{
  "family": "saplings-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "saplings",
      "image": "your-account.dkr.ecr.region.amazonaws.com/saplings:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:saplings/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/saplings",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      }
    }
  ]
}
```

### ECS Service Configuration

```yaml
# ecs-service.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ecs-service-config
data:
  service.json: |
    {
      "serviceName": "saplings-service",
      "cluster": "saplings-cluster",
      "taskDefinition": "saplings-app:1",
      "desiredCount": 3,
      "launchType": "FARGATE",
      "networkConfiguration": {
        "awsvpcConfiguration": {
          "subnets": ["subnet-12345", "subnet-67890"],
          "securityGroups": ["sg-abcdef"],
          "assignPublicIp": "DISABLED"
        }
      },
      "loadBalancers": [
        {
          "targetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/saplings/123456",
          "containerName": "saplings",
          "containerPort": 8000
        }
      ],
      "serviceTags": [
        {"key": "Environment", "value": "production"},
        {"key": "Application", "value": "saplings"}
      ]
    }
```

### Google Cloud Run Deployment

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: saplings-app
  namespace: default
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/saplings:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: saplings-secrets
              key: openai-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
```

## Environment Configuration

### Environment-Specific Settings

```python
# config/environments.py
import os
from enum import Enum
from saplings import AgentBuilder

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig:
    """Environment-specific configuration."""
    
    @staticmethod
    def get_config(env: Environment):
        configs = {
            Environment.DEVELOPMENT: {
                "log_level": "DEBUG",
                "gasa_enabled": False,
                "monitoring_enabled": True,
                "self_healing_enabled": False,
                "max_workers": 2,
                "timeout": 60,
                "rate_limit": 1000,
            },
            Environment.STAGING: {
                "log_level": "INFO",
                "gasa_enabled": True,
                "monitoring_enabled": True,
                "self_healing_enabled": True,
                "max_workers": 4,
                "timeout": 120,
                "rate_limit": 500,
            },
            Environment.PRODUCTION: {
                "log_level": "WARNING",
                "gasa_enabled": True,
                "monitoring_enabled": True,
                "self_healing_enabled": True,
                "max_workers": 8,
                "timeout": 300,
                "rate_limit": 100,
            }
        }
        return configs[env]

def create_environment_agent():
    """Create agent based on current environment."""
    env_name = os.getenv("ENVIRONMENT", "development")
    env = Environment(env_name)
    config = EnvironmentConfig.get_config(env)
    
    builder = AgentBuilder.for_openai("gpt-4o")
    
    if env == Environment.PRODUCTION:
        # Production-specific configuration
        builder = builder \
            .with_gasa_enabled(config["gasa_enabled"]) \
            .with_monitoring_enabled(config["monitoring_enabled"]) \
            .with_self_healing_enabled(config["self_healing_enabled"]) \
            .with_api_key_from_env("OPENAI_API_KEY") \
            .with_memory_path("/app/data/memory") \
            .with_rate_limiting(config["rate_limit"])
    
    elif env == Environment.STAGING:
        # Staging-specific configuration
        builder = builder \
            .with_gasa_enabled(config["gasa_enabled"]) \
            .with_monitoring_enabled(config["monitoring_enabled"]) \
            .with_self_healing_enabled(config["self_healing_enabled"]) \
            .with_memory_path("./staging_data/memory")
    
    else:
        # Development configuration
        builder = builder \
            .with_monitoring_enabled(config["monitoring_enabled"]) \
            .with_memory_path("./dev_data/memory")
    
    return builder.build()
```

### Configuration Validation

```python
# config/validator.py
import os
from typing import List, Dict, Any

class ConfigValidator:
    """Validate production configuration."""
    
    @staticmethod
    def validate_production_config() -> Dict[str, Any]:
        """Validate all required production settings."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "ENVIRONMENT",
            "DATABASE_URL",
            "REDIS_URL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        
        # Validate specific values
        environment = os.getenv("ENVIRONMENT")
        if environment not in ["development", "staging", "production"]:
            validation_result["errors"].append(
                f"Invalid ENVIRONMENT value: {environment}"
            )
        
        # Security checks
        if environment == "production":
            # Check for development keys in production
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key.startswith("sk-test") or "test" in openai_key.lower():
                validation_result["warnings"].append(
                    "Using test API key in production environment"
                )
            
            # Check for debug settings
            if os.getenv("DEBUG", "").lower() == "true":
                validation_result["warnings"].append(
                    "Debug mode enabled in production"
                )
        
        return validation_result

# Usage in application startup
if __name__ == "__main__":
    validation = ConfigValidator.validate_production_config()
    
    if not validation["valid"]:
        print("Configuration validation failed:")
        for error in validation["errors"]:
            print(f"ERROR: {error}")
        exit(1)
    
    if validation["warnings"]:
        print("Configuration warnings:")
        for warning in validation["warnings"]:
            print(f"WARNING: {warning}")
    
    # Proceed with application startup
    agent = create_environment_agent()
```

## Scaling Strategies

### Horizontal Scaling

```python
# scaling/horizontal.py
from saplings import AgentBuilder
import asyncio
import aioredis
from typing import List

class HorizontalScalingManager:
    """Manage horizontally scaled agent instances."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.agents: List[AgentBuilder] = []
        self.current_agent = 0
    
    async def initialize(self, num_agents: int = 3):
        """Initialize multiple agent instances."""
        self.redis = await aioredis.from_url(self.redis_url)
        
        for i in range(num_agents):
            agent = AgentBuilder.for_openai("gpt-4o") \
                .with_memory_path(f"./data/agent_{i}") \
                .with_instance_id(f"agent_{i}") \
                .build()
            
            self.agents.append(agent)
        
        print(f"Initialized {num_agents} agent instances")
    
    async def get_agent(self) -> AgentBuilder:
        """Get next available agent using round-robin."""
        agent = self.agents[self.current_agent]
        self.current_agent = (self.current_agent + 1) % len(self.agents)
        return agent
    
    async def get_least_busy_agent(self) -> AgentBuilder:
        """Get agent with lowest current load."""
        # Check Redis for current load metrics
        agent_loads = {}
        
        for i, agent in enumerate(self.agents):
            load_key = f"agent_load:{agent.instance_id}"
            load = await self.redis.get(load_key)
            agent_loads[i] = int(load) if load else 0
        
        # Return agent with minimum load
        min_load_agent = min(agent_loads, key=agent_loads.get)
        return self.agents[min_load_agent]
    
    async def update_agent_load(self, agent_id: str, load: int):
        """Update agent load metrics."""
        load_key = f"agent_load:{agent_id}"
        await self.redis.setex(load_key, 60, load)  # Expire after 60 seconds

# Usage
async def main():
    scaling_manager = HorizontalScalingManager("redis://localhost:6379")
    await scaling_manager.initialize(num_agents=5)
    
    # Process requests
    agent = await scaling_manager.get_least_busy_agent()
    result = await agent.run("Your task here")
```

### Vertical Scaling

```python
# scaling/vertical.py
import psutil
import asyncio
from saplings import AgentBuilder

class VerticalScalingManager:
    """Manage vertical scaling based on resource usage."""
    
    def __init__(self):
        self.agent = None
        self.current_config = "standard"
        self.monitoring = True
    
    async def create_optimized_agent(self):
        """Create agent optimized for current system resources."""
        # Get system information
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"System resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        
        if memory_gb >= 16 and cpu_count >= 8:
            # High-performance configuration
            self.agent = AgentBuilder.full_featured("openai", "gpt-4o") \
                .with_gasa_enabled(True) \
                .with_gasa_max_hops(3) \
                .with_retrieval_max_documents(50) \
                .with_planner_total_budget(10.0) \
                .build()
            self.current_config = "high_performance"
            
        elif memory_gb >= 8 and cpu_count >= 4:
            # Standard configuration
            self.agent = AgentBuilder.standard("openai", "gpt-4o") \
                .with_gasa_enabled(True) \
                .with_gasa_max_hops(2) \
                .with_retrieval_max_documents(20) \
                .build()
            self.current_config = "standard"
            
        else:
            # Minimal configuration for limited resources
            self.agent = AgentBuilder.minimal("openai", "gpt-4o") \
                .with_gasa_enabled(False) \
                .with_retrieval_max_documents(5) \
                .build()
            self.current_config = "minimal"
        
        print(f"Created agent with {self.current_config} configuration")
    
    async def monitor_and_adjust(self):
        """Monitor resource usage and adjust configuration."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            print(f"Resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
            
            # Adjust configuration based on usage
            if cpu_percent > 90 or memory_percent > 90:
                if self.current_config != "minimal":
                    print("High resource usage detected, scaling down")
                    await self.scale_down()
            
            elif cpu_percent < 30 and memory_percent < 50:
                if self.current_config == "minimal":
                    print("Low resource usage detected, scaling up")
                    await self.scale_up()
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def scale_down(self):
        """Scale down to use fewer resources."""
        if self.current_config == "high_performance":
            # Scale to standard
            await self.create_optimized_agent()  # Will create standard config
        elif self.current_config == "standard":
            # Scale to minimal
            self.agent = AgentBuilder.minimal("openai", "gpt-4o").build()
            self.current_config = "minimal"
    
    async def scale_up(self):
        """Scale up to use more resources if available."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if self.current_config == "minimal" and memory_gb >= 8:
            self.agent = AgentBuilder.standard("openai", "gpt-4o").build()
            self.current_config = "standard"
        elif self.current_config == "standard" and memory_gb >= 16:
            await self.create_optimized_agent()  # Will create high_performance config
```

## Health Checks & Monitoring

### Application Health Checks

```python
# monitoring/health.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import time
from typing import Dict, Any

app = FastAPI()

class HealthCheck(BaseModel):
    status: str
    timestamp: float
    version: str
    checks: Dict[str, Any]

class HealthChecker:
    """Comprehensive health checking for Saplings applications."""
    
    def __init__(self, agent):
        self.agent = agent
        self.start_time = time.time()
    
    async def check_agent_health(self) -> Dict[str, Any]:
        """Check agent-specific health."""
        try:
            # Test basic functionality
            start_time = time.time()
            test_result = await asyncio.wait_for(
                self.agent.run("Health check test"), 
                timeout=10
            )
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "test_result_length": len(test_result) if test_result else 0
            }
        except asyncio.TimeoutError:
            return {"status": "unhealthy", "error": "timeout"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_memory_health(self) -> Dict[str, Any]:
        """Check memory system health."""
        try:
            # Test memory operations
            test_doc = "Health check document"
            self.agent.add_document(test_doc, {"type": "health_check"})
            
            # Test retrieval
            results = await self.agent.run("Find health check documents")
            
            return {
                "status": "healthy",
                "memory_accessible": True,
                "retrieval_working": len(results) > 0 if results else False
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies."""
        checks = {}
        
        # Check API connectivity
        try:
            # Test with a simple request
            result = await asyncio.wait_for(
                self.agent.run("Say hello"), 
                timeout=5
            )
            checks["llm_api"] = {
                "status": "healthy",
                "response_received": bool(result)
            }
        except Exception as e:
            checks["llm_api"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        return checks

@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with component status."""
    # This would be injected with your actual agent instance
    # health_checker = HealthChecker(agent)
    
    checks = {
        "application": {"status": "healthy"},
        # "agent": await health_checker.check_agent_health(),
        # "memory": await health_checker.check_memory_health(),
        # "dependencies": await health_checker.check_external_dependencies(),
    }
    
    # Determine overall status
    overall_status = "healthy"
    for check in checks.values():
        if check.get("status") != "healthy":
            overall_status = "unhealthy"
            break
    
    return HealthCheck(
        status=overall_status,
        timestamp=time.time(),
        version="1.0.0",
        checks=checks
    )

@app.get("/ready")
async def readiness():
    """Readiness check for Kubernetes."""
    # Check if application is ready to receive traffic
    try:
        # Perform minimal checks
        return {"status": "ready", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Not ready")
```

### Monitoring Integration

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from typing import Optional

# Prometheus metrics
REQUEST_COUNT = Counter('saplings_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('saplings_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('saplings_active_connections', 'Active connections')
AGENT_MEMORY_USAGE = Gauge('saplings_agent_memory_bytes', 'Agent memory usage')

class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def update_active_connections(self, count: int):
        """Update active connections gauge."""
        ACTIVE_CONNECTIONS.set(count)
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage gauge."""
        AGENT_MEMORY_USAGE.set(bytes_used)

# FastAPI middleware for metrics
from fastapi import Request, Response
import time

async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

## Load Balancing

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream saplings_backend {
        least_conn;  # Use least connections algorithm
        
        # Backend servers
        server saplings-app-1:8000 max_fails=3 fail_timeout=30s;
        server saplings-app-2:8000 max_fails=3 fail_timeout=30s;
        server saplings-app-3:8000 max_fails=3 fail_timeout=30s;
        
        # Health checks
        keepalive 32;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name api.your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name api.your-domain.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        
        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        
        # Gzip compression
        gzip on;
        gzip_types text/plain application/json;
        
        location / {
            # Rate limiting
            limit_req zone=api burst=20 nodelay;
            
            # Proxy settings
            proxy_pass http://saplings_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }
        
        location /health {
            # Health check endpoint
            access_log off;
            proxy_pass http://saplings_backend;
        }
        
        location /metrics {
            # Metrics endpoint (restrict access)
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://saplings_backend;
        }
    }
}
```

### Application Load Balancer (AWS)

```json
{
  "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
  "Properties": {
    "Name": "saplings-alb",
    "Type": "application",
    "Scheme": "internet-facing",
    "SecurityGroups": ["sg-12345678"],
    "Subnets": ["subnet-12345", "subnet-67890"],
    "Tags": [
      {"Key": "Environment", "Value": "production"},
      {"Key": "Application", "Value": "saplings"}
    ]
  }
}
```

This comprehensive production deployment guide provides everything needed to deploy Saplings applications at scale with proper monitoring, security, and operational practices.