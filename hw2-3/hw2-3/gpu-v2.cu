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
int num_blocks;

int dim;
int num_bins;

int *bins;

int *bin_start;
int *bin_start_buf;

__device__ int get_bidx_from_x_y(double x, double y, int dim){
    return (int)floor(x / BINSIZE) * dim + (int)floor(y / BINSIZE);
}

__device__ void apply_force_bidir_gpu(particle_t& particle, particle_t& neighbor) {
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

    /* particle.ax += coef * dx; */
    /* particle.ay += coef * dy; */
    atomicAdd(&particle.ax, coef * dx);
    atomicAdd(&particle.ay, coef * dy);

    /* neighbor.ax -= coef * dx; */
    /* neighbor.ay -= coef * dy; */
    atomicAdd(&neighbor.ax, -coef * dx);
    atomicAdd(&neighbor.ay, -coef * dy);
}

__global__ void clear_axay(particle_t *parts, int num_parts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //parallel particles
    if(idx >= num_parts) return;
    parts[idx].ax = parts[idx].ay = 0;
}

__global__ void compute_forces_gpu(
        particle_t* parts,
        int *bins,
        int *bin_start,
        int num_bins,
        int dim){

    int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    int bidx = blockIdx.x * blockDim.x + threadIdx.x; //parallel bin
    // out of bound
    if(bidx >= num_bins) return;

    int bi = bidx / dim;
    int bj = bidx % dim;

    for(int i = bin_start[bidx]; i < bin_start[bidx+1]; ++i){
        particle_t &p1 = parts[bins[i]];
        for(int j = i+1; j < bin_start[bidx+1]; ++j){
            particle_t &p2 = parts[bins[j]];
            apply_force_bidir_gpu(p1, p2);
        }
    }

    for(int d = 5; d < 9; d++){
        int ni = bi + dir[d][0];
        int nj = bj + dir[d][1];
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
                apply_force_bidir_gpu(p1, p2);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_parts)
        return;

    particle_t* p = &particles[idx];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->x += (p->vx += p->ax * dt) * dt;
    p->y += (p->vy += p->ay * dt) * dt;

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

__global__ void count_each_bin_gpu(
        particle_t *parts, int *bin_start, int num_parts, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_parts) return;

    int bidx = get_bidx_from_x_y(parts[idx].x, parts[idx].y, dim);

    // bin_start[bidx]++
    atomicAdd(bin_start + bidx, 1);
}

__global__ void put_particle_indices_in_bin_gpu(
        particle_t *parts, int *bins, int num_parts,
        int *bin_start, int dim, int num_bins){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_parts) return;

    int bidx = get_bidx_from_x_y(parts[idx].x, parts[idx].y, dim);
    // get the old value
    int dst_idx = atomicAdd(bin_start + bidx, 1); //keep updating thru every part in the bin.
    // Put this particle's index into this bin
    bins[dst_idx] = idx;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    num_bins = dim * dim;
    num_blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // parts[bins[i]]
    cudaMalloc((void **)&bins, num_parts * sizeof(int));

    cudaMalloc((void **)&bin_start, (num_bins + 1) * sizeof(int));
    cudaMalloc((void **)&bin_start_buf, (num_bins + 1) * sizeof(int));

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Reset the count to 0
    thrust::device_ptr<int> ptr(bin_start);
    thrust::fill(ptr, ptr + num_bins + 1, 0);

    // Count how many particles in each bin
    count_each_bin_gpu<<<num_blocks, NUM_THREADS>>>(
            parts, bin_start + 1, num_parts, dim);

    // Count each bin's start position
    thrust::device_ptr<int> bin_start_wrapper(bin_start);
    thrust::inclusive_scan(bin_start_wrapper + 1, bin_start_wrapper + num_bins + 1, bin_start_wrapper + 1); //in-place scan

    // Put particles into each bin
    cudaMemcpy(bin_start_buf, bin_start, (num_bins + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    put_particle_indices_in_bin_gpu<<<num_blocks, NUM_THREADS>>>(
            parts, bins, num_parts, bin_start_buf, dim, num_bins);

    // Compute (ax, ay)
    clear_axay<<<num_blocks, NUM_THREADS>>>(parts, num_parts);

    // Each thread will be responsible for one bin
    compute_forces_gpu<<<(num_bins + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
            parts, bins, bin_start, num_bins, dim);

    // Each thread move the particle
    move_gpu<<<num_blocks, NUM_THREADS>>>(parts, num_parts, size);
}
