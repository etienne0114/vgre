package controllers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vgrev1alpha1 "vgre-operator/api/v1alpha1"
)

// VgreClusterReconciler reconciles a VgreCluster object
type VgreClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func toPtr(s string) *string {
	return &s
}

// +kubebuilder:rbac:groups=vgre.io,resources=vgreclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=vgre.io,resources=vgreclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=vgre.io,resources=vgreclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=networkpolicies,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *VgreClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	// 1. Fetch the VgreCluster instance
	var cluster vgrev1alpha1.VgreCluster
	if err := r.Get(ctx, req.NamespacedName, &cluster); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "unable to fetch VgreCluster")
		return ctrl.Result{}, err
	}

	// 2. Set default ports if unspecified
	masterPort := cluster.Spec.MasterPort
	if masterPort == 0 {
		masterPort = 7777
	}
	discoveryPort := cluster.Spec.DiscoveryPort
	if discoveryPort == 0 {
		discoveryPort = 7778
	}

	// 3. Reconcile HMAC Secret
	secretName := cluster.Spec.SecretName
	if secretName == "" {
		secretName = fmt.Sprintf("vgre-%s-auth-token", cluster.Name)
	}

	var secret corev1.Secret
	secretKey := client.ObjectKey{Namespace: cluster.Namespace, Name: secretName}
	err := r.Get(ctx, secretKey, &secret)
	if err != nil && apierrors.IsNotFound(err) {
		// Generate cryptographically secure token
		tokenBytes := make([]byte, 32)
		if _, err := rand.Read(tokenBytes); err != nil {
			log.Error(err, "failed to generate random token")
			return ctrl.Result{}, err
		}
		tokenHex := hex.EncodeToString(tokenBytes)

		secret = corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      secretName,
				Namespace: cluster.Namespace,
			},
			Data: map[string][]byte{
				"auth-token": []byte(tokenHex),
			},
		}
		if err := controllerutil.SetControllerReference(&cluster, &secret, r.Scheme); err != nil {
			log.Error(err, "failed to set controller reference on Secret")
			return ctrl.Result{}, err
		}
		log.Info("Creating new auth token secret", "Secret.Namespace", secret.Namespace, "Secret.Name", secret.Name)
		if err := r.Create(ctx, &secret); err != nil {
			log.Error(err, "failed to create Secret")
			return ctrl.Result{}, err
		}
	} else if err != nil {
		log.Error(err, "failed to get Secret")
		return ctrl.Result{}, err
	}

	// 4. Reconcile Master Service
	serviceName := fmt.Sprintf("vgre-%s-master", cluster.Name)
	var svc corev1.Service
	svcKey := client.ObjectKey{Namespace: cluster.Namespace, Name: serviceName}
	err = r.Get(ctx, svcKey, &svc)
	if err != nil && apierrors.IsNotFound(err) {
		svc = corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: cluster.Namespace,
			},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{
						Name:       "tcp-control",
						Protocol:   corev1.ProtocolTCP,
						Port:       masterPort,
						TargetPort: intstr.FromInt(int(masterPort)),
					},
					{
						Name:       "udp-discovery",
						Protocol:   corev1.ProtocolUDP,
						Port:       discoveryPort,
						TargetPort: intstr.FromInt(int(discoveryPort)),
					},
				},
				Selector: map[string]string{
					"vgre.io/cluster": cluster.Name,
					"vgre.io/role":    "master",
				},
				Type: corev1.ServiceTypeClusterIP,
			},
		}
		if err := controllerutil.SetControllerReference(&cluster, &svc, r.Scheme); err != nil {
			log.Error(err, "failed to set controller reference on Service")
			return ctrl.Result{}, err
		}
		log.Info("Creating VGRE Master Service", "Service.Namespace", svc.Namespace, "Service.Name", svc.Name)
		if err := r.Create(ctx, &svc); err != nil {
			log.Error(err, "failed to create Service")
			return ctrl.Result{}, err
		}
	} else if err != nil {
		log.Error(err, "failed to get Service")
		return ctrl.Result{}, err
	} else {
		portsUpdated := false
		if len(svc.Spec.Ports) != 2 {
			portsUpdated = true
		} else {
			if svc.Spec.Ports[0].Port != masterPort || svc.Spec.Ports[1].Port != discoveryPort {
				portsUpdated = true
			}
		}
		if portsUpdated {
			svc.Spec.Ports = []corev1.ServicePort{
				{
					Name:       "tcp-control",
					Protocol:   corev1.ProtocolTCP,
					Port:       masterPort,
					TargetPort: intstr.FromInt(int(masterPort)),
				},
				{
					Name:       "udp-discovery",
					Protocol:   corev1.ProtocolUDP,
					Port:       discoveryPort,
					TargetPort: intstr.FromInt(int(discoveryPort)),
				},
			}
			if err := r.Update(ctx, &svc); err != nil {
				log.Error(err, "failed to update Service ports")
				return ctrl.Result{}, err
			}
		}
	}

	// 5. Reconcile Master Deployment
	masterDepName := fmt.Sprintf("vgre-%s-master", cluster.Name)
	var masterDep appsv1.Deployment
	masterDepKey := client.ObjectKey{Namespace: cluster.Namespace, Name: masterDepName}
	err = r.Get(ctx, masterDepKey, &masterDep)

	masterReplicas := int32(1)
	masterPodLabels := map[string]string{
		"vgre.io/cluster": cluster.Name,
		"vgre.io/role":    "master",
	}

	masterPodSpec := corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: masterPodLabels,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "master",
					Image: cluster.Spec.MasterImage,
					Ports: []corev1.ContainerPort{
						{
							Name:          "tcp-control",
							ContainerPort: masterPort,
							Protocol:      corev1.ProtocolTCP,
						},
						{
							Name:          "udp-discovery",
							ContainerPort: discoveryPort,
							Protocol:      corev1.ProtocolUDP,
						},
					},
					Env: []corev1.EnvVar{
						{
							Name:  "VGRE_PORT",
							Value: strconv.Itoa(int(masterPort)),
						},
						{
							Name: "VGRE_TCP_AUTH_TOKEN",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: secretName,
									},
									Key: "auth-token",
								},
							},
						},
						{
							Name:  "VGRE_SHM_SUFFIX",
							Value: cluster.Name,
						},
						{
							Name: "POD_IP",
							ValueFrom: &corev1.EnvVarSource{
								FieldRef: &corev1.ObjectFieldSelector{
									FieldPath: "status.podIP",
								},
							},
						},
						{
							Name:  "VGRE_CLUSTER_ADVERTISED_ADDRESS",
							Value: fmt.Sprintf("$(POD_IP):%d", masterPort),
						},
					},
				},
			},
		},
	}

	if err != nil && apierrors.IsNotFound(err) {
		masterDep = appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      masterDepName,
				Namespace: cluster.Namespace,
			},
			Spec: appsv1.DeploymentSpec{
				Replicas: &masterReplicas,
				Selector: &metav1.LabelSelector{
					MatchLabels: masterPodLabels,
				},
				Template: masterPodSpec,
			},
		}
		if err := controllerutil.SetControllerReference(&cluster, &masterDep, r.Scheme); err != nil {
			log.Error(err, "failed to set controller reference on Master Deployment")
			return ctrl.Result{}, err
		}
		log.Info("Creating Master Deployment", "Deployment.Namespace", masterDep.Namespace, "Deployment.Name", masterDep.Name)
		if err := r.Create(ctx, &masterDep); err != nil {
			log.Error(err, "failed to create Master Deployment")
			return ctrl.Result{}, err
		}
	} else if err != nil {
		log.Error(err, "failed to get Master Deployment")
		return ctrl.Result{}, err
	} else {
		masterDep.Spec.Template.Spec.Containers[0].Image = cluster.Spec.MasterImage
		masterDep.Spec.Template.Spec.Containers[0].Ports = masterPodSpec.Spec.Containers[0].Ports
		masterDep.Spec.Template.Spec.Containers[0].Env = masterPodSpec.Spec.Containers[0].Env
		if err := r.Update(ctx, &masterDep); err != nil {
			log.Error(err, "failed to update Master Deployment")
			return ctrl.Result{}, err
		}
	}

	// 6. Reconcile Workers
	workerLabels := map[string]string{
		"vgre.io/cluster": cluster.Name,
		"vgre.io/role":    "worker",
	}

	workerPodSpec := corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: workerLabels,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "worker",
					Image: cluster.Spec.WorkerImage,
					Args: []string{
						"--port",
						strconv.Itoa(int(masterPort)),
					},
					Env: []corev1.EnvVar{
						{
							Name:  "VGRE_CLUSTER_MASTER_ADDRESS",
							Value: fmt.Sprintf("%s:%d", serviceName, masterPort),
						},
						{
							Name: "VGRE_TCP_AUTH_TOKEN",
							ValueFrom: &corev1.EnvVarSource{
								SecretKeyRef: &corev1.SecretKeySelector{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: secretName,
									},
									Key: "auth-token",
								},
							},
						},
						{
							Name:  "VGRE_SHM_SUFFIX",
							Value: cluster.Name,
						},
					},
				},
			},
		},
	}

	deploymentMode := cluster.Spec.DeploymentMode
	if deploymentMode == "" {
		deploymentMode = "Deployment"
	}

	var readyWorkers int32
	if deploymentMode == "DaemonSet" {
		var oldDep appsv1.Deployment
		oldDepKey := client.ObjectKey{Namespace: cluster.Namespace, Name: fmt.Sprintf("vgre-%s-workers", cluster.Name)}
		if err := r.Get(ctx, oldDepKey, &oldDep); err == nil {
			log.Info("Deleting old worker Deployment due to mode switch to DaemonSet", "Deployment.Name", oldDep.Name)
			_ = r.Delete(ctx, &oldDep)
		}

		dsName := fmt.Sprintf("vgre-%s-workers", cluster.Name)
		var ds appsv1.DaemonSet
		dsKey := client.ObjectKey{Namespace: cluster.Namespace, Name: dsName}
		err = r.Get(ctx, dsKey, &ds)
		if err != nil && apierrors.IsNotFound(err) {
			ds = appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dsName,
					Namespace: cluster.Namespace,
				},
				Spec: appsv1.DaemonSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: workerLabels,
					},
					Template: workerPodSpec,
				},
			}
			if err := controllerutil.SetControllerReference(&cluster, &ds, r.Scheme); err != nil {
				log.Error(err, "failed to set controller reference on Worker DaemonSet")
				return ctrl.Result{}, err
			}
			log.Info("Creating Worker DaemonSet", "DaemonSet.Namespace", ds.Namespace, "DaemonSet.Name", ds.Name)
			if err := r.Create(ctx, &ds); err != nil {
				log.Error(err, "failed to create Worker DaemonSet")
				return ctrl.Result{}, err
			}
		} else if err != nil {
			log.Error(err, "failed to get Worker DaemonSet")
			return ctrl.Result{}, err
		} else {
			ds.Spec.Template.Spec.Containers[0].Image = cluster.Spec.WorkerImage
			ds.Spec.Template.Spec.Containers[0].Args = workerPodSpec.Spec.Containers[0].Args
			ds.Spec.Template.Spec.Containers[0].Env = workerPodSpec.Spec.Containers[0].Env
			if err := r.Update(ctx, &ds); err != nil {
				log.Error(err, "failed to update Worker DaemonSet")
				return ctrl.Result{}, err
			}
			readyWorkers = ds.Status.NumberReady
		}
	} else {
		var oldDS appsv1.DaemonSet
		oldDSKey := client.ObjectKey{Namespace: cluster.Namespace, Name: fmt.Sprintf("vgre-%s-workers", cluster.Name)}
		if err := r.Get(ctx, oldDSKey, &oldDS); err == nil {
			log.Info("Deleting old worker DaemonSet due to mode switch to Deployment", "DaemonSet.Name", oldDS.Name)
			_ = r.Delete(ctx, &oldDS)
		}

		depName := fmt.Sprintf("vgre-%s-workers", cluster.Name)
		var dep appsv1.Deployment
		depKey := client.ObjectKey{Namespace: cluster.Namespace, Name: depName}
		err = r.Get(ctx, depKey, &dep)

		replicas := cluster.Spec.Replicas
		if replicas < 0 {
			replicas = 0
		}

		if err != nil && apierrors.IsNotFound(err) {
			dep = appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      depName,
					Namespace: cluster.Namespace,
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: &replicas,
					Selector: &metav1.LabelSelector{
						MatchLabels: workerLabels,
					},
					Template: workerPodSpec,
				},
			}
			if err := controllerutil.SetControllerReference(&cluster, &dep, r.Scheme); err != nil {
				log.Error(err, "failed to set controller reference on Worker Deployment")
				return ctrl.Result{}, err
			}
			log.Info("Creating Worker Deployment", "Deployment.Namespace", dep.Namespace, "Deployment.Name", dep.Name)
			if err := r.Create(ctx, &dep); err != nil {
				log.Error(err, "failed to create Worker Deployment")
				return ctrl.Result{}, err
			}
		} else if err != nil {
			log.Error(err, "failed to get Worker Deployment")
			return ctrl.Result{}, err
		} else {
			dep.Spec.Replicas = &replicas
			dep.Spec.Template.Spec.Containers[0].Image = cluster.Spec.WorkerImage
			dep.Spec.Template.Spec.Containers[0].Args = workerPodSpec.Spec.Containers[0].Args
			dep.Spec.Template.Spec.Containers[0].Env = workerPodSpec.Spec.Containers[0].Env
			if err := r.Update(ctx, &dep); err != nil {
				log.Error(err, "failed to update Worker Deployment")
				return ctrl.Result{}, err
			}
			readyWorkers = dep.Status.ReadyReplicas
		}
	}

	// 7. Reconcile NetworkPolicy
	npName := fmt.Sprintf("vgre-%s-network-policy", cluster.Name)
	var np networkingv1.NetworkPolicy
	npKey := client.ObjectKey{Namespace: cluster.Namespace, Name: npName}
	err = r.Get(ctx, npKey, &np)

	if cluster.Spec.NetworkPolicy {
		targetLabels := map[string]string{
			"vgre.io/cluster": cluster.Name,
		}
		tcpPort := intstr.FromInt(int(masterPort))
		udpPort := intstr.FromInt(int(discoveryPort))

		npSpec := networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: targetLabels,
			},
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					From: []networkingv1.NetworkPolicyPeer{
						{
							PodSelector: &metav1.LabelSelector{
								MatchLabels: targetLabels,
							},
						},
					},
					Ports: []networkingv1.NetworkPolicyPort{
						{
							Protocol: (*corev1.Protocol)(toPtr(string(corev1.ProtocolTCP))),
							Port:     &tcpPort,
						},
						{
							Protocol: (*corev1.Protocol)(toPtr(string(corev1.ProtocolUDP))),
							Port:     &udpPort,
						},
					},
				},
			},
			PolicyTypes: []networkingv1.PolicyType{
				networkingv1.PolicyTypeIngress,
			},
		}

		if err != nil && apierrors.IsNotFound(err) {
			np = networkingv1.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name:      npName,
					Namespace: cluster.Namespace,
				},
				Spec: npSpec,
			}
			if err := controllerutil.SetControllerReference(&cluster, &np, r.Scheme); err != nil {
				log.Error(err, "failed to set controller reference on NetworkPolicy")
				return ctrl.Result{}, err
			}
			log.Info("Creating VGRE NetworkPolicy", "NetworkPolicy.Namespace", np.Namespace, "NetworkPolicy.Name", np.Name)
			if err := r.Create(ctx, &np); err != nil {
				log.Error(err, "failed to create NetworkPolicy")
				return ctrl.Result{}, err
			}
		} else if err != nil {
			log.Error(err, "failed to get NetworkPolicy")
			return ctrl.Result{}, err
		} else {
			np.Spec = npSpec
			if err := r.Update(ctx, &np); err != nil {
				log.Error(err, "failed to update NetworkPolicy")
				return ctrl.Result{}, err
			}
		}
	} else {
		if err == nil {
			log.Info("Deleting VGRE NetworkPolicy", "NetworkPolicy.Namespace", np.Namespace, "NetworkPolicy.Name", np.Name)
			if err := r.Delete(ctx, &np); err != nil {
				log.Error(err, "failed to delete NetworkPolicy")
				return ctrl.Result{}, err
			}
		}
	}

	// 8. Update Status
	cluster.Status.MasterIP = svc.Spec.ClusterIP
	cluster.Status.ReadyWorkers = readyWorkers
	if cluster.Status.MasterIP != "" && readyWorkers > 0 {
		cluster.Status.Phase = "Running"
	} else {
		cluster.Status.Phase = "Pending"
	}

	if err := r.Status().Update(ctx, &cluster); err != nil {
		log.Error(err, "failed to update VgreCluster status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *VgreClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&vgrev1alpha1.VgreCluster{}).
		Owns(&appsv1.Deployment{}).
		Owns(&appsv1.DaemonSet{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.Secret{}).
		Owns(&networkingv1.NetworkPolicy{}).
		Complete(r)
}
