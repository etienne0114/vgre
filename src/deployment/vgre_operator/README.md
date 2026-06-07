# VGRE Kubernetes Operator

The VGRE Operator is a Go-based Kubernetes controller designed to automate the deployment, secure configuration, and scaling of Virtual GPU Runtime Engine (VGRE) clusters. It manages both the Master (coordinator) node and Worker workloads, ensuring cryptographically secure HMAC-SHA256 authentication and optimized network topologies.

## Key Features

- **Automated Security Handshakes**: Dynamically generates and manages a high-entropy 256-bit HMAC-SHA256 token Secret for the cluster, automatically mounting it into both Master and Worker components.
- **Flexible Worker Topologies**: Supports deploying Workers as a scaling **Deployment** (replicas-driven) or a **DaemonSet** (one worker instance per Kubernetes worker node).
- **Network Isolation**: Optionally provisions a target `NetworkPolicy` restricting ingress traffic on control ports `7777` (TCP) and discovery ports `7778` (UDP) strictly to members of the VGRE cluster.
- **IPC Isolation**: Configures cluster-isolated Shared Memory (SHM) regions via the `VGRE_SHM_SUFFIX` variable.

## CRD Specification (`VgreCluster`)

```yaml
apiVersion: vgre.io/v1alpha1
kind: VgreCluster
metadata:
  name: prod-vgre-cluster
spec:
  replicas: 4                  # Number of worker instances (ignored if DaemonSet)
  deploymentMode: Deployment   # "Deployment" or "DaemonSet"
  masterImage: vgre-master:v1  # Master image containing vgre_dashboard/master
  workerImage: vgre-worker:v1  # Worker image containing vgre-worker C++ daemon
  masterPort: 7777             # Control port (default: 7777)
  discoveryPort: 7778          # UDP discovery port (default: 7778)
  networkPolicy: true          # Provision ingress control network policy
```

## Reconciled Resources

1. **Secret**: `vgre-<cluster-name>-auth-token` containing the SHA256 hex key.
2. **Service**: `vgre-<cluster-name>-master` exposing `tcp-control` (TCP) and `udp-discovery` (UDP).
3. **Deployment**: `vgre-<cluster-name>-master` (1 replica) hosting the Master Coordinator.
4. **DaemonSet / Deployment**: `vgre-<cluster-name>-workers` executing the compute node worker daemons.
5. **NetworkPolicy**: `vgre-<cluster-name>-network-policy` locking down communication between VGRE components.

## Master Node: Headless Mode

The master Deployment uses `vgre-worker --is-master` (not `--master`) as its ENTRYPOINT. This starts the coordinator in headless container mode: binds on the cluster port, accepts worker connections, does not launch the Flutter dashboard. The Dockerfile at `docker/Dockerfile.master` is configured correctly.

## Local Development and Building

### Prerequisites

- Go 1.22+
- A running Kubernetes cluster (or `kind`/`minikube` for testing)

### Build the Operator Manager

```bash
go build -o bin/manager main.go
```

### Run Tests

Execute the unit tests verifying the reconciliation logic:

```bash
go test -v ./...
```
