package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VgreClusterSpec defines the desired state of VgreCluster
type VgreClusterSpec struct {
	// Replicas is the desired number of VGRE Worker instances (pods).
	// Ignored if DeploymentMode is "DaemonSet".
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas,omitempty"`

	// DeploymentMode specifies the workload kind for the VGRE Workers.
	// Can be "DaemonSet" (one worker per node) or "Deployment" (replica count).
	// Default is "Deployment".
	// +kubebuilder:validation:Enum=Deployment;DaemonSet
	DeploymentMode string `json:"deploymentMode,omitempty"`

	// MasterImage is the container image to use for the VGRE Master.
	MasterImage string `json:"masterImage"`

	// WorkerImage is the container image to use for the VGRE Workers.
	WorkerImage string `json:"workerImage"`

	// MasterPort is the port used for control and data connection (default: 7777).
	MasterPort int32 `json:"masterPort,omitempty"`

	// DiscoveryPort is the UDP port used for discovery (default: 7778).
	DiscoveryPort int32 `json:"discoveryPort,omitempty"`

	// SecretName is the name of the Secret to hold the VGRE auth token.
	// If empty, the operator will generate a Secret name and generate a secure token.
	SecretName string `json:"secretName,omitempty"`

	// NetworkPolicy specifies whether the operator should provision a default NetworkPolicy
	// restricting communication strictly to VGRE cluster ports.
	NetworkPolicy bool `json:"networkPolicy,omitempty"`
}

// VgreClusterStatus defines the observed state of VgreCluster
type VgreClusterStatus struct {
	// Phase is the current execution phase: Pending, Running, Failed.
	Phase string `json:"phase,omitempty"`

	// ReadyWorkers is the number of ready worker pods.
	ReadyWorkers int32 `json:"readyWorkers,omitempty"`

	// MasterIP is the internal cluster IP address of the Master Service.
	MasterIP string `json:"masterIP,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// VgreCluster is the Schema for the vgreclusters API
type VgreCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VgreClusterSpec   `json:"spec,omitempty"`
	Status VgreClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VgreClusterList contains a list of VgreCluster
type VgreClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VgreCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VgreCluster{}, &VgreClusterList{})
}
