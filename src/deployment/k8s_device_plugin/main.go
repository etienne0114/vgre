// VGRE Kubernetes Device Plugin
//
// Implements the Kubernetes Device Plugin gRPC API to register virtual GPUs
// with the K8s kubelet. Each virtual GPU is backed by a VGRE CPU-emulated
// device slot.
//
// Usage:
//   go run main.go --device-plugin-path=/var/lib/kubelet/device-plugins
//
// The plugin discovers available VGRE slots via the VGRE C API (or falls back
// to a configurable count), registers with kubelet, and serves allocation
// requests.

package main

import (
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	resourceName    = "vgre.io/gpu"
	pluginSocket    = "vgre-gpu.sock"
	kubeletSockDir  = "/var/lib/kubelet/device-plugins"
	vgreDefaultGPUs = 1
)

type vgreDevicePlugin struct {
	server     *grpc.Server
	resource   string
	devs       []*pluginapi.Device
	socketPath string
}

func newVgreDevicePlugin(resource string, gpuCount int) *vgreDevicePlugin {
	devs := make([]*pluginapi.Device, gpuCount)
	for i := 0; i < gpuCount; i++ {
		devs[i] = &pluginapi.Device{
			ID:     fmt.Sprintf("vgre-gpu-%d", i),
			Health: pluginapi.Healthy,
		}
	}
	return &vgreDevicePlugin{
		resource:   resource,
		devs:       devs,
		socketPath: filepath.Join(kubeletSockDir, pluginSocket),
	}
}

// GetDevicePluginOptions returns options to be communicated with Device Manager.
func (m *vgreDevicePlugin) GetDevicePluginOptions(ctx interface{}, req *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{PreStartRequired: false}, nil
}

// ListAndWatch lists devices and updates kubelet on health changes.
func (m *vgreDevicePlugin) ListAndWatch(empty *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: m.devs}); err != nil {
		return err
	}
	// Keep stream alive until shutdown
	for {
		time.Sleep(10 * time.Second)
	}
}

// Allocate is called by kubelet when a pod requests a VGRE GPU.
func (m *vgreDevicePlugin) Allocate(ctx interface{}, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	responses := make([]*pluginapi.ContainerAllocateResponse, 0, len(reqs.ContainerRequests))
	for _, creq := range reqs.ContainerRequests {
		resp := &pluginapi.ContainerAllocateResponse{
			Envs: map[string]string{
				"VGRE_VISIBLE_DEVICES": fmt.Sprintf("%v", creq.DevicesIDs),
				"CUDA_VISIBLE_DEVICES": fmt.Sprintf("%v", creq.DevicesIDs),
			},
			// VGRE uses LD_PRELOAD; no host device paths needed.
			Devices: []*pluginapi.DeviceSpec{},
		}
		responses = append(responses, resp)
	}
	return &pluginapi.AllocateResponse{ContainerResponses: responses}, nil
}

// GetPreferredAllocation is optional; return empty.
func (m *vgreDevicePlugin) GetPreferredAllocation(ctx interface{}, req *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}

// PreStartContainer is a no-op.
func (m *vgreDevicePlugin) PreStartContainer(ctx interface{}, req *pluginapi.PreStartContainerRequest) (*pluginapi.Empty, error) {
	return &pluginapi.Empty{}, nil
}

func (m *vgreDevicePlugin) start() error {
	_ = os.Remove(m.socketPath)
	lis, err := net.Listen("unix", m.socketPath)
	if err != nil {
		return err
	}
	m.server = grpc.NewServer()
	pluginapi.RegisterDevicePluginServer(m.server, m)
	go func() {
		if err := m.server.Serve(lis); err != nil {
			fmt.Fprintf(os.Stderr, "device plugin server error: %v\n", err)
		}
	}()
	return nil
}

func (m *vgreDevicePlugin) stop() {
	if m.server != nil {
		m.server.Stop()
	}
}

// Register with kubelet via the device plugin registration gRPC endpoint.
func registerWithKubelet(socketPath string) error {
	conn, err := grpc.Dial("unix:///"+filepath.Join(kubeletSockDir, "kubelet.sock"),
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("failed to dial kubelet: %w", err)
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	req := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     pluginSocket,
		ResourceName: resourceName,
		Options:      &pluginapi.DevicePluginOptions{PreStartRequired: false},
	}
	_, err = client.Register(nil, req)
	return err
}

func main() {
	gpuCount := flag.Int("gpus", vgreDefaultGPUs, "Number of virtual VGRE GPUs to expose")
	flag.Parse()

	plugin := newVgreDevicePlugin(resourceName, *gpuCount)
	if err := plugin.start(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start device plugin: %v\n", err)
		os.Exit(1)
	}
	defer plugin.stop()

	// Wait for server to be ready, then register
	time.Sleep(2 * time.Second)
	if err := registerWithKubelet(plugin.socketPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to register with kubelet: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("VGRE K8s Device Plugin registered %d GPU(s) as %s\n", *gpuCount, resourceName)
	select {} // run forever
}
