# VGRE Kubernetes Device Plugin

Exposes VGRE virtual GPUs to Kubernetes via the [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) API.

## Quick Start

```bash
# Build container image
docker build -t vgre-device-plugin:latest .

# Deploy as a DaemonSet on every node
kubectl apply -f daemonset.yaml
```

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--gpus` | `1` | Number of virtual VGRE GPUs to expose per node |

## Pod Usage

Request a VGRE GPU in a pod spec:

```yaml
resources:
  limits:
    vgre.io/gpu: "1"
```

The plugin sets `VGRE_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` in the container environment so the VGRE runtime intercepts CUDA calls.

## Architecture

- The plugin implements `DevicePlugin` gRPC service.
- It registers with kubelet at `/var/lib/kubelet/device-plugins/kubelet.sock`.
- Allocation responses contain environment variables only (no host device files needed for CPU emulation).
