#include <immintrin.h>

const char* dgemm_desc = "2-level blocked & loop reordered & repacking dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE_L1 40
#endif
#define BLOCK_SIZE_L2 BLOCK_SIZE_L1*4 
#define MK_SIZE 4
#define MK_K 16

#define min(a, b) (((a) < (b)) ? (a) : (b))


static inline void micro_kernel(int eM, int mK, double* A, double* B, double* C) {
    // TODO: use SIMD 
    for (int k = 0; k <  mK; ++k) {   
        for (int ii=0; ii<MK_SIZE; ++ii){
            for (int jj=0; jj<MK_SIZE;++jj){
                C[ii+jj*eM] += A[ii+k*MK_SIZE] * B[jj+k*MK_SIZE];
            }
        }             
    }
}

static void do_block(int lda, int M, int N, int K, int eM, int eN, double* A, double* B, double* C, double* rC) {

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {         
            rC[i + j*eM] = C[i + j*lda]; //((i<lda) & (j<lda)) ? C[i + j*lda]:0;
        }
    }

    // For each row i of A
    for (int k =0; k<K; k+=MK_K){
        for (int i = 0; i < eM; i+=MK_SIZE) {
            // For each column j of B
            for (int j = 0; j < eN; j+=MK_SIZE) {
                // Compute C(i,j)
                /*
                for (int k = 0; k <  K; ++k) {   
                    for (int ii=0; ii<MK_SIZE; ++ii){
                        for (int jj=0; jj<MK_SIZE;++jj){
                            rC[i+ii+(j+jj)*eM] += A[ii+i*K+k*MK_SIZE] * B[jj+j*K+k*MK_SIZE];
                        }
                    }             
                }
                */
                int mK = min(MK_K, K-k);
                micro_kernel(eM, mK, A+i*K+k*MK_SIZE, B+j*K+k*MK_SIZE, rC+i+j*eM);
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
    


    for (int i=0; i<M; i+=MK_SIZE) {
        for (int k = 0; k < K; ++k) {   
            for (int ii=0; ii<MK_SIZE; ++ii){
                rA[ii+i*K+k*MK_SIZE] = (ii+i <lda) ? A[ii+i+ k * lda]:0;
            }      
        }
    }

    for (int j = 0; j < N; j+=MK_SIZE) {
        for (int k = 0; k < K; ++k) {     
            for (int jj=0; jj<MK_SIZE; ++jj){
                rB[jj+j*K+k*MK_SIZE] = (j+jj<lda) ? B[k + (j+jj) * lda]:0;
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

    //double *rA, *rB, *rC ;
    //posix_memalign((void **)&rA, 64, BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double));
    //posix_memalign((void **)&rB, 64, BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double));
    //posix_memalign((void **)&rC, 64, BLOCK_SIZE_L1 * BLOCK_SIZE_L1 * sizeof(double));

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
                                int eN = ( N%MK_SIZE !=0) ? N+MK_SIZE-N%MK_SIZE:N;
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