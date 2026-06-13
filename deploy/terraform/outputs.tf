output "namespace" {
  description = "Namespace VGRE was deployed into"
  value       = var.namespace
}

output "release_name" {
  description = "Helm release name"
  value       = helm_release.vgre.name
}

output "gpu_resource" {
  description = "Kubernetes resource name to request an emulated GPU"
  value       = "vgre.io/gpu"
}
