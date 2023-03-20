#include "common.h"
#include <cuda.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <vector>
#include <stdio.h>
#include <assert.h>

#define BINSIZE (cutoff * 2.1)
#define NUM_THREADS 1024
#define MIN(x,y) (((x)<(y))?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
int blks;
int directions[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

int dim;

int *bins;
int *bin_start;
int *bin_start_buf;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(
        particle_t* parts,
        int *bins,
        int *bin_start,
        int dim){

    int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    // out of bound
    if(bidx >= dim * dim) return;

    int bi = bidx / dim;
    int bj = bidx % dim;

    for(int d = 0; d < 9; d++){
        int ni = bi + directions[d][0];
        int nj = bj + directions[d][1];
        if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
            continue;

        int nidx = ni * dim + nj;

        // Loop over its particles
        for(int i = bin_start[bidx]; i < bin_start[bidx+1]; ++i){
            particle_t &p1 = parts[bins[i]];
            // Loop over neighbor grid's particles
            for(int j = bin_start[nidx]; j < bin_start[nidx+1]; ++j){
                particle_t &p2 = parts[bins[j]];
                // Apply force to p1
                apply_force_gpu(p1, p2);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    dim = floor(size / BINSIZE) + 1;

    // parts[bins[i]]
    cudaMalloc((void **)&bins, num_parts * sizeof(int));

    cudaMalloc((void **)&bin_start, (dim * dim + 1) * sizeof(int));
    cudaMalloc((void **)&bin_start_buf, (dim * dim + 1) * sizeof(int));

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Reset the count to 0
    thrust::device_ptr<int> ptr(bin_start);
    thrust::fill(ptr, ptr + dim * dim + 1, 0);

    // Count how many particles in each bin
    // count_each_bin_gpu<<<blks, NUM_THREADS>>>(
    //         parts, bin_start + 1, num_parts, dim);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= num_parts) return;

    int bidx = floor(parts[tid].x / BINSIZE) * dim + floor(parts[tid].y / BINSIZE);

    // bin_start[bidx]++
    atomicAdd(bin_start + 1 + bidx, 1);
    

    // Count each bin's start position
    thrust::device_ptr<int> bin_start_wrapper(bin_start);
    thrust::inclusive_scan(bin_start_wrapper + 1, bin_start_wrapper + dim * dim + 1, bin_start_wrapper + 1);

    // Put particles into each bin
    cudaMemcpy(bin_start_buf, bin_start, (dim * dim + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    // get the old value
    int dst_idx = atomicAdd(bin_start_buf + bidx, 1);
    // Put this particle's index into this bin
    bins[dst_idx] = tid;
    
    parts[tid].ax = parts[tid].ay = 0;

    // Each thread will be responsible for one bin
    compute_forces_gpu<<<(dim * dim + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
            parts, bins, bin_start, dim);

    // Each thread move the particle
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
