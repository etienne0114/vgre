package controllers

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	vgrev1alpha1 "vgre-operator/api/v1alpha1"
)

func newTestScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	_ = appsv1.AddToScheme(s)
	_ = networkingv1.AddToScheme(s)
	_ = vgrev1alpha1.AddToScheme(s)
	return s
}

func TestVgreClusterController(t *testing.T) {
	scheme := newTestScheme()

	cluster := &vgrev1alpha1.VgreCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster",
			Namespace: "default",
		},
		Spec: vgrev1alpha1.VgreClusterSpec{
			Replicas:       2,
			DeploymentMode: "Deployment",
			MasterImage:    "vgre-master:latest",
			WorkerImage:    "vgre-worker:latest",
			MasterPort:     7777,
			DiscoveryPort:  7778,
			NetworkPolicy:  true,
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster).
		WithStatusSubresource(&vgrev1alpha1.VgreCluster{}).
		Build()

	r := &VgreClusterReconciler{
		Client: cl,
		Scheme: scheme,
	}

	req := reconcile.Request{
		NamespacedName: types.NamespacedName{
			Name:      "test-cluster",
			Namespace: "default",
		},
	}
	res, err := r.Reconcile(context.TODO(), req)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	if res.Requeue {
		t.Fatal("reconcile requested requeue")
	}

	// Verify Secret was created
	secret := &corev1.Secret{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-auth-token", Namespace: "default"}, secret)
	if err != nil {
		t.Fatalf("failed to find Secret: %v", err)
	}
	tokenBytes := secret.Data["auth-token"]
	if len(tokenBytes) == 0 {
		t.Fatal("auth-token key is missing from Secret or empty")
	}

	// Verify Service was created
	svc := &corev1.Service{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-master", Namespace: "default"}, svc)
	if err != nil {
		t.Fatalf("failed to find Service: %v", err)
	}
	if svc.Spec.Ports[0].Port != 7777 || svc.Spec.Ports[1].Port != 7778 {
		t.Errorf("unexpected ports on Master Service: %+v", svc.Spec.Ports)
	}

	// Verify Master Deployment was created
	masterDep := &appsv1.Deployment{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-master", Namespace: "default"}, masterDep)
	if err != nil {
		t.Fatalf("failed to find Master Deployment: %v", err)
	}
	if masterDep.Spec.Template.Spec.Containers[0].Image != "vgre-master:latest" {
		t.Errorf("unexpected image on Master container: %s", masterDep.Spec.Template.Spec.Containers[0].Image)
	}

	// Verify Worker Deployment was created
	workerDep := &appsv1.Deployment{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-workers", Namespace: "default"}, workerDep)
	if err != nil {
		t.Fatalf("failed to find Worker Deployment: %v", err)
	}
	if *workerDep.Spec.Replicas != 2 {
		t.Errorf("unexpected replicas count on Worker Deployment: %d", *workerDep.Spec.Replicas)
	}

	// Verify NetworkPolicy was created
	np := &networkingv1.NetworkPolicy{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-network-policy", Namespace: "default"}, np)
	if err != nil {
		t.Fatalf("failed to find NetworkPolicy: %v", err)
	}
	if np.Spec.PodSelector.MatchLabels["vgre.io/cluster"] != "test-cluster" {
		t.Errorf("unexpected pod selector on NetworkPolicy: %+v", np.Spec.PodSelector)
	}
}

func TestVgreClusterControllerDaemonSet(t *testing.T) {
	scheme := newTestScheme()

	cluster := &vgrev1alpha1.VgreCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-cluster-ds",
			Namespace: "default",
		},
		Spec: vgrev1alpha1.VgreClusterSpec{
			DeploymentMode: "DaemonSet",
			MasterImage:    "vgre-master:latest",
			WorkerImage:    "vgre-worker:latest",
			MasterPort:     7777,
			DiscoveryPort:  7778,
			NetworkPolicy:  false,
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster).
		WithStatusSubresource(&vgrev1alpha1.VgreCluster{}).
		Build()

	r := &VgreClusterReconciler{
		Client: cl,
		Scheme: scheme,
	}

	req := reconcile.Request{
		NamespacedName: types.NamespacedName{
			Name:      "test-cluster-ds",
			Namespace: "default",
		},
	}
	_, err := r.Reconcile(context.TODO(), req)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	// Verify Worker DaemonSet was created
	ds := &appsv1.DaemonSet{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-ds-workers", Namespace: "default"}, ds)
	if err != nil {
		t.Fatalf("failed to find Worker DaemonSet: %v", err)
	}
	if ds.Spec.Template.Spec.Containers[0].Image != "vgre-worker:latest" {
		t.Errorf("unexpected image on Worker container: %s", ds.Spec.Template.Spec.Containers[0].Image)
	}

	// Verify NetworkPolicy was NOT created (networkPolicy=false)
	np := &networkingv1.NetworkPolicy{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-test-cluster-ds-network-policy", Namespace: "default"}, np)
	if err == nil {
		t.Error("NetworkPolicy should not exist when networkPolicy=false")
	}
}

func TestVgreClusterControllerDefaultPorts(t *testing.T) {
	scheme := newTestScheme()

	// Test with zero ports — should default to 7777/7778
	cluster := &vgrev1alpha1.VgreCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "defaults-cluster",
			Namespace: "default",
		},
		Spec: vgrev1alpha1.VgreClusterSpec{
			Replicas:    1,
			MasterImage: "vgre-master:latest",
			WorkerImage: "vgre-worker:latest",
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(cluster).
		WithStatusSubresource(&vgrev1alpha1.VgreCluster{}).
		Build()

	r := &VgreClusterReconciler{
		Client: cl,
		Scheme: scheme,
	}

	req := reconcile.Request{
		NamespacedName: types.NamespacedName{
			Name:      "defaults-cluster",
			Namespace: "default",
		},
	}
	_, err := r.Reconcile(context.TODO(), req)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	svc := &corev1.Service{}
	err = cl.Get(context.TODO(), types.NamespacedName{Name: "vgre-defaults-cluster-master", Namespace: "default"}, svc)
	if err != nil {
		t.Fatalf("failed to find Service: %v", err)
	}
	if svc.Spec.Ports[0].Port != 7777 {
		t.Errorf("expected default master port 7777, got %d", svc.Spec.Ports[0].Port)
	}
	if svc.Spec.Ports[1].Port != 7778 {
		t.Errorf("expected default discovery port 7778, got %d", svc.Spec.Ports[1].Port)
	}
}
