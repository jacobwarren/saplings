# Production Deployment Guide

This guide covers deploying Saplings applications in production environments, including containerization, scaling strategies, and operational best practices.

## Table of Contents

- [Production Architecture](#production-architecture)
- [Containerization](#containerization)
- [Environment Configuration](#environment-configuration)
- [Security Considerations](#security-considerations)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling Strategies](#scaling-strategies)
- [Operational Best Practices](#operational-best-practices)

## Production Architecture

### Application Structure

```python
# production_app.py
import os
import logging
from saplings import Agent, AgentBuilder, AgentConfig
from saplings.api.memory import MemoryConfig
from saplings.api.tool_factory import ToolFactoryConfig, SecurityLevel

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_production_agent():
    """Create a production-ready agent with secure configuration."""
    
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "MEMORY_PATH", "OUTPUT_DIR"]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Required environment variable {var} not set")
    
    # Production configuration
    config = AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        memory_path=os.getenv("MEMORY_PATH", "/app/memory"),
        output_dir=os.getenv("OUTPUT_DIR", "/app/output"),
        enable_monitoring=True,
        enable_gasa=False,  # Disable for production unless needed
        memory_config=MemoryConfig.secure(),
        tool_factory_config=ToolFactoryConfig(
            security_level=SecurityLevel.HIGH,
            enable_sandboxing=True,
            timeout_seconds=30
        )
    )
    
    return Agent(config)

def main():
    """Main application entry point."""
    try:
        agent = create_production_agent()
        logging.info("Production agent initialized successfully")
        
        # Your application logic here
        # e.g., web server, task queue consumer, etc.
        
    except Exception as e:
        logging.error(f"Failed to initialize production agent: {e}")
        raise

if __name__ == "__main__":
    main()
```

## Containerization

### Dockerfile

```dockerfile
# Multi-stage build for production efficiency
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 saplings

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/memory /app/output /app/logs && \
    chown -R saplings:saplings /app

# Copy application code
COPY --chown=saplings:saplings . .

# Switch to non-root user
USER saplings

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import saplings; print('OK')" || exit 1

# Expose application port
EXPOSE 8000

# Default command
CMD ["python", "production_app.py"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  saplings-app:
    build: 
      context: .
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MEMORY_PATH=/app/memory
      - OUTPUT_DIR=/app/output
      - LOG_LEVEL=INFO
    volumes:
      - ./memory:/app/memory
      - ./output:/app/output
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import saplings; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add monitoring services
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: saplings-production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: saplings-config
  namespace: saplings-production
data:
  LOG_LEVEL: "INFO"
  MEMORY_PATH: "/app/memory"
  OUTPUT_DIR: "/app/output"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: saplings-secrets
  namespace: saplings-production
type: Opaque
data:
  # Base64 encoded API keys
  OPENAI_API_KEY: <base64-encoded-key>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saplings-app
  namespace: saplings-production
  labels:
    app: saplings-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: saplings-app
  template:
    metadata:
      labels:
        app: saplings-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: saplings-app
        image: your-registry/saplings-app:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: saplings-config
        - secretRef:
            name: saplings-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: memory-storage
          mountPath: /app/memory
        - name: output-storage
          mountPath: /app/output
      volumes:
      - name: memory-storage
        persistentVolumeClaim:
          claimName: saplings-memory-pvc
      - name: output-storage
        persistentVolumeClaim:
          claimName: saplings-output-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: saplings-service
  namespace: saplings-production
spec:
  selector:
    app: saplings-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: saplings-memory-pvc
  namespace: saplings-production
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: saplings-output-pvc
  namespace: saplings-production
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

## Environment Configuration

### Environment Variables

```bash
# .env.production
# API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Application Configuration
LOG_LEVEL=INFO
MEMORY_PATH=/app/memory
OUTPUT_DIR=/app/output
MAX_MEMORY_SIZE=1GB
MAX_OUTPUT_SIZE=5GB

# Security Configuration
SECURITY_LEVEL=HIGH
ENABLE_SANDBOXING=true
SANDBOX_TIMEOUT=30

# Monitoring Configuration
ENABLE_MONITORING=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080

# Database (if using external storage)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### Configuration Management

```python
# config/production.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # API Configuration
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # Application Settings
    log_level: str = "INFO"
    memory_path: str = "/app/memory"
    output_dir: str = "/app/output"
    max_memory_size: str = "1GB"
    
    # Security Settings
    security_level: str = "HIGH"
    enable_sandboxing: bool = True
    sandbox_timeout: int = 30
    
    # Performance Settings
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    
    # Monitoring Settings
    enable_monitoring: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            memory_path=os.environ.get("MEMORY_PATH", "/app/memory"),
            output_dir=os.environ.get("OUTPUT_DIR", "/app/output"),
            max_memory_size=os.environ.get("MAX_MEMORY_SIZE", "1GB"),
            security_level=os.environ.get("SECURITY_LEVEL", "HIGH"),
            enable_sandboxing=os.environ.get("ENABLE_SANDBOXING", "true").lower() == "true",
            sandbox_timeout=int(os.environ.get("SANDBOX_TIMEOUT", "30")),
            max_concurrent_requests=int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "60")),
            enable_monitoring=os.environ.get("ENABLE_MONITORING", "true").lower() == "true",
            metrics_port=int(os.environ.get("METRICS_PORT", "9090")),
            health_check_port=int(os.environ.get("HEALTH_CHECK_PORT", "8080"))
        )

# Usage
config = ProductionConfig.from_env()
```

## Security Considerations

### Secure Agent Configuration

```python
from saplings.api.tool_factory import SecurityLevel, SandboxType
from saplings.api.memory import MemoryConfig, PrivacyLevel

def create_secure_production_agent():
    """Create agent with maximum security for production."""
    
    # Secure memory configuration
    memory_config = MemoryConfig.secure()
    memory_config.secure_store.privacy_level = PrivacyLevel.HASH_AND_DP
    memory_config.secure_store.hash_salt = os.environ.get("MEMORY_SALT", "production-salt")
    
    # Tool factory security
    tool_config = ToolFactoryConfig(
        security_level=SecurityLevel.HIGH,
        sandbox_type=SandboxType.DOCKER,
        enable_sandboxing=True,
        timeout_seconds=10,  # Short timeout for production
        docker_image="python:3.11-slim"
    )
    
    config = AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
        memory_config=memory_config,
        tool_factory_config=tool_config,
        enable_monitoring=True
    )
    
    return Agent(config)
```

### Network Security

```python
# network_security.py
import asyncio
import aiohttp
from aiohttp import web

class SecurityMiddleware:
    """Security middleware for web applications."""
    
    async def __call__(self, request, handler):
        # Add security headers
        response = await handler(request)
        
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response

# Rate limiting
class RateLimitMiddleware:
    """Simple rate limiting middleware."""
    
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    async def __call__(self, request, handler):
        client_ip = request.remote
        now = asyncio.get_event_loop().time()
        
        # Clean old entries
        self.requests = {
            ip: reqs for ip, reqs in self.requests.items()
            if any(req_time > now - self.window for req_time in reqs)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[client_ip]
                if req_time > now - self.window
            ]
            if len(recent_requests) >= self.max_requests:
                raise web.HTTPTooManyRequests()
            
            self.requests[client_ip] = recent_requests + [now]
        else:
            self.requests[client_ip] = [now]
        
        return await handler(request)
```

## Monitoring & Observability

### Health Checks

```python
# health.py
from aiohttp import web
import json
import time

class HealthChecker:
    """Application health checker."""
    
    def __init__(self, agent):
        self.agent = agent
        self.start_time = time.time()
    
    async def health_check(self, request):
        """Basic health check endpoint."""
        try:
            # Check if agent is responsive
            # Simple check - you might want to add more comprehensive checks
            status = {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "version": "1.0.0"
            }
            
            return web.json_response(status)
        
        except Exception as e:
            status = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
            return web.json_response(status, status=503)
    
    async def readiness_check(self, request):
        """Readiness check for Kubernetes."""
        try:
            # Check if all dependencies are available
            # e.g., database connections, external APIs, etc.
            
            checks = {
                "agent_initialized": self.agent is not None,
                "memory_accessible": True,  # Add actual memory check
                "api_keys_configured": True  # Add actual API key validation
            }
            
            if all(checks.values()):
                return web.json_response({"status": "ready", "checks": checks})
            else:
                return web.json_response(
                    {"status": "not_ready", "checks": checks}, 
                    status=503
                )
        
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)}, 
                status=503
            )

# Create health check routes
def setup_health_routes(app, agent):
    """Setup health check routes."""
    health_checker = HealthChecker(agent)
    
    app.router.add_get('/health', health_checker.health_check)
    app.router.add_get('/ready', health_checker.readiness_check)
```

### Metrics Collection

```python
# metrics.py
import time
import psutil
from collections import defaultdict

class MetricsCollector:
    """Collect application metrics."""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.response_times = []
        self.error_count = defaultdict(int)
        self.start_time = time.time()
    
    def record_request(self, method, path, response_time, status_code):
        """Record request metrics."""
        self.request_count[f"{method}_{path}"] += 1
        self.response_times.append(response_time)
        
        if status_code >= 400:
            self.error_count[status_code] += 1
    
    def get_metrics(self):
        """Get current metrics."""
        process = psutil.Process()
        
        return {
            "system": {
                "uptime": time.time() - self.start_time,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
            },
            "process": {
                "memory_info": process.memory_info()._asdict(),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
            },
            "application": {
                "total_requests": sum(self.request_count.values()),
                "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "error_rate": sum(self.error_count.values()) / sum(self.request_count.values()) if self.request_count else 0,
                "request_breakdown": dict(self.request_count),
                "error_breakdown": dict(self.error_count),
            }
        }

# Metrics middleware
class MetricsMiddleware:
    """Collect metrics for each request."""
    
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
    
    async def __call__(self, request, handler):
        start_time = time.time()
        
        try:
            response = await handler(request)
            response_time = time.time() - start_time
            
            self.metrics.record_request(
                request.method,
                request.path,
                response_time,
                response.status
            )
            
            return response
        
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(
                request.method,
                request.path,
                response_time,
                500
            )
            raise
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: saplings-hpa
  namespace: saplings-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: saplings-app
  minReplicas: 3
  maxReplicas: 10
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
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
```

### Load Balancing

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: saplings-ingress
  namespace: saplings-production
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - saplings.yourdomain.com
    secretName: saplings-tls
  rules:
  - host: saplings.yourdomain.com
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

## Operational Best Practices

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_production_logging():
    """Configure production logging."""
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler for container logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for local development
    if os.path.exists('/app/logs'):
        file_handler = RotatingFileHandler(
            '/app/logs/saplings.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
```

### Error Handling & Recovery

```python
# error_handling.py
import asyncio
import logging
from typing import Callable, Any

class ProductionErrorHandler:
    """Production-grade error handling."""
    
    def __init__(self, max_retries=3, base_delay=1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
    
    async def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(
                        f"Function {func.__name__} failed after {self.max_retries} retries: {e}"
                    )
                    break
                
                delay = self.base_delay * (2 ** attempt)
                self.logger.warning(
                    f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def handle_exception(self, e: Exception, context: str = ""):
        """Handle exceptions with appropriate logging and recovery."""
        error_msg = f"Error in {context}: {type(e).__name__}: {e}"
        
        if isinstance(e, (ConnectionError, TimeoutError)):
            self.logger.warning(f"Transient error: {error_msg}")
        else:
            self.logger.error(f"Application error: {error_msg}", exc_info=True)
```

### Resource Management

```python
# resource_management.py
import os
import psutil
import asyncio
from contextlib import asynccontextmanager

class ResourceManager:
    """Manage system resources for production deployment."""
    
    def __init__(self, max_memory_percent=80, max_cpu_percent=90):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.logger = logging.getLogger(__name__)
    
    def check_system_resources(self):
        """Check if system resources are within limits."""
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if memory_percent > self.max_memory_percent:
            self.logger.warning(f"High memory usage: {memory_percent}%")
            return False
        
        if cpu_percent > self.max_cpu_percent:
            self.logger.warning(f"High CPU usage: {cpu_percent}%")
            return False
        
        return True
    
    @asynccontextmanager
    async def resource_limits(self):
        """Context manager for resource-limited execution."""
        try:
            if not self.check_system_resources():
                raise ResourceError("System resources exhausted")
            
            yield
            
        finally:
            # Cleanup resources if needed
            pass

# Usage in production application
async def process_with_limits(agent, task):
    """Process task with resource limits."""
    resource_manager = ResourceManager()
    
    async with resource_manager.resource_limits():
        return await agent.run(task)
```

This production deployment guide provides comprehensive coverage of deploying Saplings applications in production environments, focusing on the actual containerization and sandboxing capabilities available in the framework.