const char* dgemm_desc = "Two level blocked dgemm.";

#ifndef BLOCK_SIZE1
#define BLOCK_SIZE1 64
#endif
#ifndef BLOCK_SIZE2
#define BLOCK_SIZE2 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int k = 0; k < K; ++k) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            for (int i = 0; i < M; ++i) {
                C[i + j * lda] += A[i + k * lda] * B[k + j * lda];
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE1) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE1) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE1) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE1, lda - i);
                int N = min(BLOCK_SIZE1, lda - j);
                int K = min(BLOCK_SIZE1, lda - k);
                for (int ii = 0; ii < M; ii += BLOCK_SIZE2){
                    for (int jj = 0; jj < N; jj += BLOCK_SIZE2){
                        for (int kk = 0; kk < K; kk += BLOCK_SIZE2) {
                            int MM = min(BLOCK_SIZE2, M - ii);
                            int NN = min(BLOCK_SIZE2, N - jj);
                            int KK = min(BLOCK_SIZE2, K - kk);
                            do_block(lda, MM, NN, KK, A + (i+ii) + (k+kk) * lda, B + (k+kk) + (j+jj) * lda, C + i+ii + (j+jj) * lda);
                        }
                    }
                }
            }
        }
    }
}