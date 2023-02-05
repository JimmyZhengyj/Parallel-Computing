#include <immintrin.h>

const char* dgemm_desc = "2-level blocked & loop reordered & repacking dgemm & microkernel.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE_L1 48*4
#endif
#define BLOCK_SIZE_L2 BLOCK_SIZE_L1*4 
#define MK_SIZE 4
#define MK_SIZE_2 12
#define MK_K 48

#define min(a, b) (((a) < (b)) ? (a) : (b))


static inline void micro_kernel(int eM, int mK, double* A, double* B, double* C) {
    __m256d c0 = _mm256_load_pd(C);
    __m256d c1 = _mm256_load_pd(C+eM);
    __m256d c2 = _mm256_load_pd(C+2*eM);
    __m256d c3 = _mm256_load_pd(C+3*eM);
    __m256d c4 = _mm256_load_pd(C+4*eM);
    __m256d c5 = _mm256_load_pd(C+5*eM);
    __m256d c6 = _mm256_load_pd(C+6*eM);
    __m256d c7 = _mm256_load_pd(C+7*eM);
    __m256d c8 = _mm256_load_pd(C+8*eM);
    __m256d c9 = _mm256_load_pd(C+9*eM);
    __m256d c10 = _mm256_load_pd(C+10*eM);
    __m256d c11 = _mm256_load_pd(C+11*eM);
    __m256d b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11;
    __m256d a;
    for (int k = 0; k <  mK; ++k) {
        b0 = _mm256_set1_pd(B[k*MK_SIZE_2]);
        b1 = _mm256_set1_pd(B[k*MK_SIZE_2+1]);
        b2 = _mm256_set1_pd(B[k*MK_SIZE_2+2]);
        b3 = _mm256_set1_pd(B[k*MK_SIZE_2+3]);
        b4 = _mm256_set1_pd(B[k*MK_SIZE_2+4]);
        b5 = _mm256_set1_pd(B[k*MK_SIZE_2+5]);
        b6 = _mm256_set1_pd(B[k*MK_SIZE_2+6]);
        b7 = _mm256_set1_pd(B[k*MK_SIZE_2+7]);
        b8 = _mm256_set1_pd(B[k*MK_SIZE_2+8]);
        b9 = _mm256_set1_pd(B[k*MK_SIZE_2+9]);
        b10 = _mm256_set1_pd(B[k*MK_SIZE_2+10]);
        b11 = _mm256_set1_pd(B[k*MK_SIZE_2+11]);
        a  = _mm256_load_pd(A+k*MK_SIZE);
        c0 = _mm256_fmadd_pd(a, b0, c0);
        c1 = _mm256_fmadd_pd(a, b1, c1);
        c2 = _mm256_fmadd_pd(a, b2, c2);
        c3 = _mm256_fmadd_pd(a, b3, c3);
        c4 = _mm256_fmadd_pd(a, b4, c4);
        c5 = _mm256_fmadd_pd(a, b5, c5);
        c6 = _mm256_fmadd_pd(a, b6, c6);
        c7 = _mm256_fmadd_pd(a, b7, c7);
        c8 = _mm256_fmadd_pd(a, b8, c8);
        c9 = _mm256_fmadd_pd(a, b9, c9);
        c10 = _mm256_fmadd_pd(a, b10, c10);
        c11 = _mm256_fmadd_pd(a, b11, c11);
    }
    _mm256_store_pd(C, c0);
    _mm256_store_pd(C+eM,c1);
    _mm256_store_pd(C+2*eM,c2);
    _mm256_store_pd(C+3*eM,c3);
    _mm256_store_pd(C+4*eM,c4);
    _mm256_store_pd(C+5*eM,c5);
    _mm256_store_pd(C+6*eM,c6);
    _mm256_store_pd(C+7*eM,c7);
    _mm256_store_pd(C+8*eM,c8);
    _mm256_store_pd(C+9*eM,c9);
    _mm256_store_pd(C+10*eM,c10);
    _mm256_store_pd(C+11*eM,c11);
}

static void do_block(int lda, int M, int N, int K, int eM, int eN, double* A, double* B, double* C, double* rC) {

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {         
            rC[i + j*eM] = C[i + j*lda];
        }
    }

    for (int k =0; k<K; k+=MK_K){
        int mK = min(MK_K, K-k);
        for (int i = 0; i < eM; i+=MK_SIZE) {
            for (int j = 0; j < eN; j+=MK_SIZE_2) {
                //micro_kernel(eM, mK, A+i*K+k*MK_SIZE, B+j*K+k*MK_SIZE, rC+i+j*eM);
                micro_kernel(eM, mK, A+i*MK_K+k*eM, B+j*MK_K+k*eN, rC+i+j*eM);
            }
        }
    }
    
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {         
            C[i + j*lda] = rC[i + j*eM];
        }
    }
}

static void copy_block(int lda, int M, int N, int K, double* A, double* B, double* rA, double* rB) {
    
    for (int k = 0; k < K; k+=MK_K){
        int mK = min(MK_K, K-k);
        for (int i=0; i<M; i+=MK_SIZE)  {   
            for (int kk=0; kk< mK; ++kk){
                //rA[ii+i*K+k*MK_SIZE] = (ii+i <lda) ? A[ii+i+ k * lda]:0;
                int minMk = min(MK_SIZE, lda-i);
                for (int ii=0; ii<minMk; ++ii) {
                    rA[ii+kk*MK_SIZE+i*MK_K+k*M] = A[ii+i+ (k+kk) * lda]; //(ii+i <lda) ? A[ii+i+ (k+kk) * lda]:0;
                }
                for (int ii=minMk; ii< MK_SIZE; ++ii){
                    rA[ii+kk*MK_SIZE+i*MK_K+k*M] = 0;
                }
            }      
        }
    }

    for (int k = 0; k < K; k+=MK_K) {
        int mK = min(MK_K, K-k);
        for (int j = 0; j < N; j+=MK_SIZE_2) {     
            for (int kk=0; kk< mK; ++kk){
                //rB[jj+j*K+k*MK_SIZE] = (j+jj<lda) ? B[k + (j+jj) * lda]:0;
                int minNk = min(MK_SIZE_2, lda-j);
                for (int jj=0; jj<minNk; ++jj){
                    rB[jj+kk*MK_SIZE_2+j*MK_K+k*N] = B[k + kk + (j+jj) * lda]; //(j+jj<lda) ? B[k + kk + (j+jj) * lda]:0;
                }
                for (int jj=minNk; jj < MK_SIZE_2; ++jj){
                    rB[jj+kk*MK_SIZE_2+j*MK_K+k*N] = 0;
                }
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.                                                                       
 * On exit, A and B maintain their input values. */

void square_dgemm(int lda, double* A, double* B, double* C) {

    double *rA = _mm_malloc(BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double), 64);
    double *rB = _mm_malloc(BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double), 64);
    double *rC = _mm_malloc(BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double), 32);

    // For each block-row of A
    for (int k_l2 = 0; k_l2 < lda; k_l2 += BLOCK_SIZE_L2) {
        // For each block-column of B
        for (int j_l2 = 0; j_l2 < lda; j_l2 += BLOCK_SIZE_L2) {
            // Accumulate block dgemms into block of C
            for (int i_l2 = 0; i_l2 < lda; i_l2 += BLOCK_SIZE_L2) {
                for (int k = k_l2; k < k_l2 + min(BLOCK_SIZE_L2,lda-k_l2); k += BLOCK_SIZE_L1) {
                    for (int j = j_l2; j < j_l2 + min(BLOCK_SIZE_L2,lda-j_l2); j += BLOCK_SIZE_L1) {
                        for (int i = i_l2; i < i_l2 + min(BLOCK_SIZE_L2,lda-i_l2); i += BLOCK_SIZE_L1) {
                            // Correct block dimensions if block "goes off edge of" the matrix
                            int M = min(BLOCK_SIZE_L1, lda - i);
                            int N = min(BLOCK_SIZE_L1, lda - j);
                            int K = min(BLOCK_SIZE_L1, lda - k);
                            int eM = ( M%MK_SIZE !=0) ? M+MK_SIZE-M%MK_SIZE:M;
                            int eN = ( N%MK_SIZE_2 !=0) ? N+MK_SIZE_2-N%MK_SIZE_2:N;
                            // Perform individual block dgemm
                            copy_block(lda, eM, eN, K, A + i + k * lda, B + k + j * lda, rA, rB);
                            do_block(lda, M, N, K,eM,eN, rA, rB, C + i + j * lda, rC);
                        }
                    }
                }
            }
        }
    }

    _mm_free(rA);
    _mm_free(rB);
    _mm_free(rC);
}