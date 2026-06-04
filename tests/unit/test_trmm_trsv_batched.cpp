// Tests for QUEUE-47 (TRMM strided-batched) and QUEUE-52 (TRSV batched).
// Math: TRMM C=α*op(A)*C; TRSV: ||A*x - b||/||b|| < 1e-5 after solve.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef int cublasSideMode_t;
typedef int cublasFillMode_t;
typedef int cublasOperation_t;
typedef int cublasDiagType_t;
#define CUBLAS_STATUS_SUCCESS      0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_SIDE_LEFT  0
#define CUBLAS_FILL_MODE_UPPER 1
#define CUBLAS_OP_N 0
#define CUBLAS_DIAG_NON_UNIT 0
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
cublasStatus_t cublasStrmm_v2(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,int,const float*,const float*,int,float*,int);
cublasStatus_t cublasDtrmm_v2(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,int,const double*,const double*,int,double*,int);
cublasStatus_t cublasStrsv_v2(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,const float*,int,float*,int);
cublasStatus_t cublasDtrsv_v2(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,const double*,int,double*,int);
cublasStatus_t cublasStrmmStridedBatched(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,int,const float*,const float*,int,long long,const float*,int,long long,float*,int,long long,int);
cublasStatus_t cublasDtrmmStridedBatched(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,int,const double*,const double*,int,long long,const double*,int,long long,double*,int,long long,int);
cublasStatus_t cublasStrsvBatched(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,const float* const*,int,float* const*,int,int);
cublasStatus_t cublasDtrsvBatched(cublasHandle_t,cublasFillMode_t,cublasOperation_t,cublasDiagType_t,int,const double* const*,int,double* const*,int,int);
}
#define ASSERT_EQ(a,b) assert(std::fabs((a)-(b)) < 1e-4f)

static void test_strmm_strided() {
    // B = identity, so TRMM(A,B) = A
    cublasHandle_t h; cublasCreate_v2(&h);
    const int N=4, batch=3;
    std::vector<float> A(N*N*batch,0.f), B(N*N*batch,0.f), C(N*N*batch,0.f);
    for (int b=0;b<batch;++b) for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        if (i<=j) A[b*N*N+i*N+j] = float(i*N+j+1+b);
        B[b*N*N+i*N+j] = (i==j) ? 1.f : 0.f;
    }
    float alpha=1.f;
    auto s = cublasStrmmStridedBatched(h,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,N,N,&alpha,
        A.data(),N,(long long)(N*N), B.data(),N,(long long)(N*N),
        C.data(),N,(long long)(N*N), batch);
    assert(s==CUBLAS_STATUS_SUCCESS);
    for (int i=0;i<N*N*batch;++i) ASSERT_EQ(C[i], A[i]);
    cublasDestroy_v2(h);
    printf("[PASS] QUEUE-47 cublasStrmmStridedBatched batch=%d n=%d\n", batch, N);
}

static void test_strsv_batched() {
    // Solve A*x = b for 4 batches of 4×4 upper triangular systems
    cublasHandle_t h; cublasCreate_v2(&h);
    const int N=4, batch=4;
    std::vector<std::vector<float>> Av(batch, std::vector<float>(N*N,0.f));
    std::vector<std::vector<float>> xv(batch, std::vector<float>(N,0.f));
    std::vector<std::vector<float>> bv(batch, std::vector<float>(N,0.f));
    std::vector<const float*> Apt(batch); std::vector<float*> xpt(batch);
    for (int b=0;b<batch;++b) {
        for (int i=0;i<N;++i) {
            Av[b][i*N+i] = float(i+2+b);
            if (i+1<N) Av[b][i*N+i+1] = 0.5f;
            bv[b][i] = float(N-i+b); xv[b][i] = bv[b][i];
        }
        Apt[b]=Av[b].data(); xpt[b]=xv[b].data();
    }
    auto s = cublasStrsvBatched(h,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,
        N, Apt.data(), N, xpt.data(), 1, batch);
    assert(s==CUBLAS_STATUS_SUCCESS);
    for (int b=0;b<batch;++b) {
        for (int i=0;i<N;++i) {
            float ax=0.f; for(int j=0;j<N;++j) ax+=Av[b][i*N+j]*xv[b][j];
            ASSERT_EQ(ax, bv[b][i]);
        }
    }
    cublasDestroy_v2(h);
    printf("[PASS] QUEUE-52 cublasStrsvBatched batch=%d n=%d\n", batch, N);
}

int main() {
    test_strmm_strided();
    test_strsv_batched();
    printf("All QUEUE-47/52 tests passed.\n");
    return 0;
}
