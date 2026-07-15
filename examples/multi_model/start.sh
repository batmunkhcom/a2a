#!/usr/bin/env bash
# multi_model mesh demo — 3-plugin mesh with cross-model projection
set -e

echo "Starting Multi-Model Mesh (summarizer → normalizer → mixer)..."
echo "Model-A (768-dim) → Projection → Model-B (1024-dim)"
echo "gRPC server: localhost:50051"
echo "Metrics: http://localhost:9090/metrics"
echo ""

cd "$(dirname "$0")"

python -c "
from a2a.config.loader import load_config
from a2a.runtime import A2ARuntime

config = load_config('a2a.yaml')
print(f'Mesh ID: {config.mesh_id}')
print(f'Models: {list(config.models.keys())}')
print(f'Plugins: {list(config.plugins.keys())}')
print(f'Routes: {len(config.routes)} configured')
print('')

# Show model dimensions
for name, m in config.models.items():
    proj = m.projection
    if proj:
        print(f'  {name}: {m.hidden_dim}-dim ({m.family}) → projection: {proj.variant}')
    else:
        print(f'  {name}: {m.hidden_dim}-dim ({m.family})')

runtime = A2ARuntime(config)
runtime.start()
print('')
print('Multi-model mesh running. Press Ctrl+C to stop.')
try:
    import time
    while runtime.is_running:
        time.sleep(1)
except KeyboardInterrupt:
    print('')
    runtime.stop()
    print('Mesh stopped.')
"
