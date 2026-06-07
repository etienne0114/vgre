// VGRE Kubernetes Device Plugin
//
// Implements the Kubernetes Device Plugin gRPC API to register virtual GPUs
// with the K8s kubelet. Each virtual GPU is backed by a VGRE CPU-emulated
// device slot.
//
// Usage:
//   go run main.go --gpus=4
//
// The plugin registers with kubelet via the Unix socket in VGRE_DEVICE_PLUGIN_PATH
// (default: /var/lib/kubelet/device-plugins), advertises <gpus> virtual GPU slots,
// and serves Allocate requests by injecting VGRE_VISIBLE_DEVICES into the container
// environment.

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	resourceName        = "vgre.io/gpu"
	pluginSocket        = "vgre-gpu.sock"
	defaultKubeletSockDir = "/var/lib/kubelet/device-plugins"
	vgreDefaultGPUs     = 1
)

// kubeletSockDir returns the device-plugin socket directory.
// Reads VGRE_DEVICE_PLUGIN_PATH first; falls back to the K8s default.
func kubeletSockDir() string {
	if v := os.Getenv("VGRE_DEVICE_PLUGIN_PATH"); v != "" {
		return v
	}
	return defaultKubeletSockDir
}

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
		socketPath: filepath.Join(kubeletSockDir(), pluginSocket),
	}
}

// GetDevicePluginOptions returns options to be communicated with Device Manager.
func (m *vgreDevicePlugin) GetDevicePluginOptions(_ context.Context, _ *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{PreStartRequired: false}, nil
}

// ListAndWatch lists devices and streams health updates to kubelet.
// Returns when the stream context is cancelled (pod deletion / kubelet restart).
func (m *vgreDevicePlugin) ListAndWatch(_ *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: m.devs}); err != nil {
		return err
	}
	// Block until the kubelet cancels the stream (pod deleted, plugin restart, etc.).
	<-stream.Context().Done()
	return nil
}

// Allocate is called by kubelet when a pod requests a VGRE GPU.
func (m *vgreDevicePlugin) Allocate(_ context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
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

// GetPreferredAllocation is optional — return empty response.
func (m *vgreDevicePlugin) GetPreferredAllocation(_ context.Context, _ *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}

// PreStartContainer is a no-op for VGRE.
func (m *vgreDevicePlugin) PreStartContainer(_ context.Context, _ *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
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

// registerWithKubelet registers this plugin with kubelet via its registration socket.
func registerWithKubelet() error {
	kubeletSock := filepath.Join(kubeletSockDir(), "kubelet.sock")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.NewClient(
		"unix:///"+kubeletSock,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to kubelet: %w", err)
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	req := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     pluginSocket,
		ResourceName: resourceName,
		Options:      &pluginapi.DevicePluginOptions{PreStartRequired: false},
	}
	_, err = client.Register(ctx, req)
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

	// Brief pause to let the gRPC server accept connections before registering.
	time.Sleep(1 * time.Second)

	if err := registerWithKubelet(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to register with kubelet: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("VGRE K8s Device Plugin registered %d GPU(s) as %s\n", *gpuCount, resourceName)
	select {} // run forever; ListAndWatch streams handle pod lifecycle
}
