variable "kubeconfig" {
  description = "Path to the kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "namespace" {
  description = "Namespace to deploy VGRE into"
  type        = string
  default     = "kube-system"
}

variable "gpus_per_node" {
  description = "Emulated GPUs advertised per node by the device plugin"
  type        = number
  default     = 1
}

variable "operator_replicas" {
  description = "Number of operator replicas"
  type        = number
  default     = 1
}

variable "device_plugin_image" {
  description = "Device plugin container image"
  type        = string
  default     = "vgre-device-plugin:latest"
}

variable "operator_image" {
  description = "Operator container image"
  type        = string
  default     = "vgre-operator:latest"
}
