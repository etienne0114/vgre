provider "kubernetes" {
  config_path = var.kubeconfig
}

provider "helm" {
  kubernetes {
    config_path = var.kubeconfig
  }
}

# Deploy the VGRE Helm chart (device plugin + operator + RBAC).
resource "helm_release" "vgre" {
  name             = "vgre"
  namespace        = var.namespace
  create_namespace = true
  chart            = "${path.module}/../helm/vgre"

  set {
    name  = "namespace"
    value = var.namespace
  }
  set {
    name  = "devicePlugin.gpusPerNode"
    value = var.gpus_per_node
  }
  set {
    name  = "devicePlugin.image.repository"
    value = split(":", var.device_plugin_image)[0]
  }
  set {
    name  = "operator.replicas"
    value = var.operator_replicas
  }
  set {
    name  = "operator.image.repository"
    value = split(":", var.operator_image)[0]
  }
}
