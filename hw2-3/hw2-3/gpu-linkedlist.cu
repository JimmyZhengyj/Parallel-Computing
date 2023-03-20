#include "common.h"
#include <cuda.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <vector>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>

#define BINSIZE (cutoff * 1.6)
#define NUM_THREADS 1024
#define MIN(x,y) (((x)<(y))?(x):(y))

/* #define P */

// Put any static global variables here that you will use throughout the simulation.
int num_blocks;

int dim;
int num_bins;

int *bin_start;
int *bin_count;
int *next;

#ifdef P
int count = 0;
auto start_time = std::chrono::steady_clock::now();
auto end_time = std::chrono::steady_clock::now();
std::chrono::duration<double> diff;

double init_time = 0;
double prepare_time = 0;
double compute_time = 0;
double move_time = 0;
#endif

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

__global__ void compute_forces_gpu(
        particle_t* parts,
        int *bin_start,
        int *bin_count,
        int *next,
        int num_bins,
        int dim){

    int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    // out of bound
    if(bidx >= num_bins) return;

    int bi = bidx / dim;
    int bj = bidx % dim;

    const int count = bin_count[bidx];
    int nei_count;

    for(int i = 0, pi = bin_start[bidx]; i < count; ++i){
        particle_t &p1 = parts[pi];

        for(int j = i+1, pj = next[pi]; j < count; ++j){
            particle_t &p2 = parts[pj];
            apply_force_bidir_gpu(p1, p2);

            // move to the next particle idx
            pj = next[pj];
        }

        // move to the next particle idx
        pi = next[pi];
    }

    for(int d = 5; d < 9; d++){
        int ni = bi + dir[d][0];
        int nj = bj + dir[d][1];
        if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
            continue;

        int nidx = ni * dim + nj;

        // Loop over its particles
        for(int i = 0, pi = bin_start[bidx]; i < count; ++i){
            particle_t &p1 = parts[pi];

            // Loop over neighbor grid's particles
            nei_count = bin_count[nidx];
            for(int j = 0, pj = bin_start[nidx]; j < nei_count; ++j){
                particle_t &p2 = parts[pj];
                // Apply force to p1
                apply_force_bidir_gpu(p1, p2);

                // move to the next particle also in this bin
                pj = next[pj];
            }

            // move to the next particle also in this bin
            pi = next[pi];
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

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    num_bins = dim * dim;
    num_blocks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // bin_start[i] = points to the last particle in this bin
    cudaMalloc((void **)&bin_start, num_bins * sizeof(int));
    cudaMalloc((void **)&bin_count, num_bins * sizeof(int));

    // next[particle idx] = the next particle also in my bin
    cudaMalloc((void **)&next, num_parts * sizeof(int));
}

__global__ void prepare(
        particle_t *parts, int num_parts,
        int *next, int *bin_start, int *bin_count, int num_bins, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_parts) return;

    int bidx = get_bidx_from_x_y(parts[idx].x, parts[idx].y, dim);

    // Let the bin point to this particle(me)
    // And point this particle to the old pointer in bin_start
    next[idx] = atomicExch(&bin_start[bidx], idx);

    // Increment it
    // NOTE: the reason we need bin_count[...] is
    // we don't want to keep checking if next[idx] == -1 (the end of the linkedlist)
    atomicAdd(&bin_count[bidx], 1);

    // also clear its axay
    parts[idx].ax = parts[idx].ay = 0;
}
__global__ void initialize(
        int *bin_start, int *bin_count, int num_bins){
    int bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(bidx >= num_bins) return;
    bin_start[bidx] = -1; // point to null
    bin_count[bidx] = 0; // each bin has zero particles at first
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
#ifdef P
    count++;
    start_time = std::chrono::steady_clock::now();
#endif

    initialize<<<(num_bins + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
            bin_start, bin_count, num_bins);

#ifdef P
    cudaDeviceSynchronize();
    //search~
    
    end_time = std::chrono::steady_clock::now();
    diff = end_time - start_time;
    init_time += diff.count();

    // Build the linkedlist
    start_time = std::chrono::steady_clock::now();
#endif

    prepare<<<num_blocks, NUM_THREADS>>>(parts, num_parts,
            next, bin_start, bin_count, num_bins, dim);

#ifdef P
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    diff = end_time - start_time;
    prepare_time += diff.count();


    // Each thread will be responsible for one bin
    start_time = std::chrono::steady_clock::now();
#endif

    compute_forces_gpu<<<(num_bins + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
            parts, bin_start, bin_count, next, num_bins, dim);

#ifdef P
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    diff = end_time - start_time;
    compute_time += diff.count();

    // Each thread move the particle
    start_time = std::chrono::steady_clock::now();
#endif

    move_gpu<<<num_blocks, NUM_THREADS>>>(parts, num_parts, size);

#ifdef P
    cudaDeviceSynchronize();
    end_time = std::chrono::steady_clock::now();
    diff = end_time - start_time;
    move_time += diff.count();

    if(count == nsteps){
        std::cout << "Init time = " << init_time << std::endl;
        std::cout << "Prepare time = " << prepare_time << std::endl;
        std::cout << "Compute time = " << compute_time << std::endl;
        std::cout << "Move time = " << move_time << std::endl;
    }
#endif
}
