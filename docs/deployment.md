# Deployment Guide

## Docker

### Single Node

```dockerfile
FROM python:3.11-slim

RUN pip install a2a-protocol

COPY a2a.yaml /etc/a2a/a2a.yaml
COPY plugins/ /opt/a2a/plugins/

EXPOSE 50051 8080 9090

CMD ["a2a", "serve", "--config", "/etc/a2a/a2a.yaml"]
```

```bash
docker build -t a2a-mesh .
docker run -p 50051:50051 -p 8080:8080 a2a-mesh
```

### Multi-Stage Build (with GPU)

```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
RUN pip install --user a2a-protocol[ml]

# Stage 2: Runtime
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY --from=builder /root/.local /root/.local
COPY a2a.yaml /etc/a2a/a2a.yaml
COPY plugins/ /opt/a2a/plugins/
ENV PATH=/root/.local/bin:$PATH
CMD ["a2a", "serve"]
```

```bash
docker build -t a2a-mesh:gpu -f Dockerfile.gpu .
docker run --gpus all -p 50051:50051 a2a-mesh:gpu
```

---

## Kubernetes

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-mesh
  labels:
    app: a2a
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a2a
  template:
    metadata:
      labels:
        app: a2a
    spec:
      containers:
        - name: a2a
          image: ghcr.io/batmunkhcom/a2a:latest
          ports:
            - containerPort: 50051
              name: grpc
            - containerPort: 8080
              name: health
            - containerPort: 9090
              name: metrics
          env:
            - name: A2A_JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: a2a-secrets
                  key: jwt-secret
          volumeMounts:
            - name: config
              mountPath: /etc/a2a
            - name: certs
              mountPath: /etc/a2a/certs
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
      volumes:
        - name: config
          configMap:
            name: a2a-config
        - name: certs
          secret:
            secretName: a2a-certs
---
apiVersion: v1
kind: Service
metadata:
  name: a2a-mesh
spec:
  selector:
    app: a2a
  ports:
    - name: grpc
      port: 50051
    - name: health
      port: 8080
    - name: metrics
      port: 9090
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: a2a-config
data:
  a2a.yaml: |
    version: "0.1"
    mesh_id: "production-mesh"
    server:
      host: "0.0.0.0"
      port: 50051
    security:
      jwt_secret: "${A2A_JWT_SECRET}"
      tls:
        enabled: true
        cert_file: "/etc/a2a/certs/server.crt"
        key_file: "/etc/a2a/certs/server.key"
        ca_file: "/etc/a2a/certs/ca.crt"
    models:
      base:
        name: "base-model"
        family: "llama"
        dtype: "fp16"
        hidden_dim: 4096
    plugins:
      worker:
        module: a2a.plugins.log_reader.plugin
        agent_id: worker
        model: base
---
apiVersion: v1
kind: Secret
metadata:
  name: a2a-secrets
type: Opaque
stringData:
  jwt-secret: "your-secret-key-here"
```

### With GPU

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

---

## Systemd (Linux)

```ini
# /etc/systemd/system/a2a.service
[Unit]
Description=A2A Protocol Mesh
After=network.target

[Service]
Type=simple
User=a2a
Group=a2a
WorkingDirectory=/opt/a2a
ExecStart=/opt/a2a/.venv/bin/a2a serve --config /etc/a2a/a2a.yaml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now a2a
```

---

## Health Endpoints

| Endpoint | Purpose | Response |
|---|---|---|
| `GET /health` | Overall status | `{"status":"ok","plugins_loaded":3,"uptime":3600}` |
| `GET /health/live` | Liveness probe | `{"alive":true}` |
| `GET /health/ready` | Readiness probe | `{"ready":true}` |

---

## Monitoring

Prometheus metrics available at `/metrics` (port 9090):

| Metric | Type | Labels |
|---|---|---|
| `a2a_tensors_sent_total` | Counter | `agent_id`, `label` |
| `a2a_tensors_received_total` | Counter | `agent_id`, `label` |
| `a2a_tensor_latency_seconds` | Histogram | `agent_id`, `label` |
| `a2a_plugins_active` | Gauge | — |
| `a2a_rate_limited_total` | Counter | `agent_id` |
| `a2a_errors_total` | Counter | `agent_id`, `code` |

---

## Scaling

### Horizontal

- Each A2A instance is independent; route by mesh_id or plugin type
- Use gRPC load balancing (client-side or proxy)
- Stateless plugins scale infinitely; stateful plugins need session affinity

### Vertical

- Increase `server.max_workers` for more concurrent tensor processing
- GPU memory: ~2× model memory (model + projection + buffers)
- FP16 halves memory, BF16 provides better dynamic range

---

## Security Checklist

- [ ] JWT secret stored in Kubernetes Secret / HashiCorp Vault
- [ ] mTLS enabled in production
- [ ] `mesh_whitelist` configured
- [ ] Rate limiting enabled per agent
- [ ] Health endpoints not exposed publicly
- [ ] gRPC port firewalled (internal only)
- [ ] Regular secret rotation
