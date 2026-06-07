# VGRE Kubernetes Device Plugin

Exposes VGRE virtual GPUs to Kubernetes via the [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) API (v1beta1).

## Quick Start

```bash
# Build container image (distroless, non-root)
docker build -t vgre-device-plugin:latest .

# Deploy as a DaemonSet on every node
kubectl apply -f daemonset.yaml
```

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--gpus` | `1` | Number of virtual VGRE GPUs to expose per node |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VGRE_DEVICE_PLUGIN_PATH` | `/var/lib/kubelet/device-plugins/kubelet.sock` | Override kubelet socket path (useful in non-standard K8s distributions) |

## Pod Usage

Request a VGRE GPU in a pod spec:

```yaml
resources:
  limits:
    vgre.io/gpu: "1"
```

The plugin sets `VGRE_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` in the container environment so the VGRE runtime intercepts CUDA calls.

## Architecture

- The plugin implements the `v1beta1.DevicePluginServer` gRPC interface.
- All gRPC handler methods accept a proper `context.Context` as the first argument.
- Registration uses `grpc.NewClient` + `insecure.NewCredentials` with a 10-second context timeout.
- `ListAndWatch` exits cleanly via `<-stream.Context().Done()` when the kubelet cancels the stream (no infinite sleep).
- Kubelet socket path is read from `VGRE_DEVICE_PLUGIN_PATH` env var at start-up; defaults to `/var/lib/kubelet/device-plugins/kubelet.sock`.
- Allocation responses contain environment variables only — no host device files needed for CPU emulation.
- Container image uses `gcr.io/distroless/static-debian12:nonroot` (minimal attack surface, non-root).

## Building

```bash
go mod download
go build -trimpath -ldflags="-s -w" -o vgre-device-plugin main.go
```

Requires Go 1.22+.
