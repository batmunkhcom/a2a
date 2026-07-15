#!/usr/bin/env bash
# basic_mesh demo — start a 2-plugin A2A mesh
set -e

echo "Starting Basic Mesh (repeater + inverter)..."
echo "gRPC server: localhost:50051"
echo "Health: http://localhost:8080/health"
echo ""

# Start from the current directory with the local a2a.yaml
cd "$(dirname "$0")"

# Use Python directly (assumes a2a is installed)
python -c "
from a2a.config.loader import load_config
from a2a.runtime import A2ARuntime

config = load_config('a2a.yaml')
print(f'Mesh ID: {config.mesh_id}')
print(f'Plugins: {list(config.plugins.keys())}')
print(f'Models: {list(config.models.keys())}')

runtime = A2ARuntime(config)
runtime.start()
print('Basic mesh running. Press Ctrl+C to stop.')
try:
    import time
    while runtime.is_running:
        time.sleep(1)
except KeyboardInterrupt:
    print('')
    runtime.stop()
    print('Mesh stopped.')
"
