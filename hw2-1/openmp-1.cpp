#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <unordered_set>

using namespace std;

// Put any static global variables here that you will use throughout the simulation.
#define BINSIZE (cutoff * 2.3)
#define NUMTHREADS 20
#define CHUNKSIZE 100

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}

vector<unordered_set<particle_t* >> bins;

int griddim;

// Ensure loop over row by row
int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
	
#pragma omp atomic
    particle.ax += coef * dx;
#pragma omp atomic
    particle.ay += coef * dy;

#pragma omp atomic
    neighbor.ax -= coef * dx;
#pragma omp atomic
    neighbor.ay -= coef * dy;
}

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    const int orig_bin = bi(p.x) * griddim + bj(p.y);

    // Update vx, vy and directly use there return value to update x,y
    p.vx    += p.ax * dt;
    p.vy    += p.ay * dt;
    p.x     += p.vx * dt;
    p.y     += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    const int new_bin = bi(p.x) * griddim + bj(p.y);

	if(orig_bin == new_bin) return;

	//update the bin
	//lock orig_bin and new_bin
#pragma omp critical
    {
        bins[orig_bin].erase(&p);
        bins[new_bin].insert(&p);
    }
    return;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
	omp_set_num_threads(NUMTHREADS);

	griddim = floor(size / BINSIZE) + 1;

	int num_bins = griddim * griddim;

    bins = vector<unordered_set<particle_t*>>(num_bins);

	//Pre-reserve the memory at once
	const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
#pragma omp for schedule(static)
	for(int i = 0; i< num_bins; ++i){
		bins[i].reserve(space);
	}

    // Put particles into the bins
#pragma omp for schedule(dynamic, CHUNKSIZE)
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;
        bins[bi(x) * griddim + bj(y)].insert(&parts[i]);
    }

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Write this function
	// Reset the acceleration
#pragma omp for schedule(dynamic, CHUNKSIZE)
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }

	    // Loop over each grid (better locality)
    unordered_set<particle_t* > :: iterator cur, neighbor, grid_begin, grid_end, neigh_grid_begin, neigh_grid_end;

#pragma omp for private(cur, neighbor, grid_begin, grid_end, neigh_grid_begin, neigh_grid_end) schedule(dynamic) collapse(2)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            int bin_index = i * griddim + j;
            // Loop over each particle in this grid
            // For each neighbor grid,
            // apply the force to the particles inside this grid
            // NOTE: the inner loop must start from l+1,
            // otherwise, we will compute the force twice for particles
            // in the same grid

            grid_begin  = bins[bin_index].begin();
            grid_end    = bins[bin_index].end();

            for(cur = grid_begin; cur != grid_end; ++cur){
                for(neighbor = next(cur); neighbor != grid_end; ++neighbor){
                    apply_force(**cur, **neighbor);
                }
            }

            for(int d = 5; d < 9; ++d){
                int bi_nei = i + dir[d][0];
                int bj_nei = j + dir[d][1];
                // out of bound
                if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                    continue;
                // Apply forces to all the particles inside this grid
                int nei_gird_index = bi_nei * griddim + bj_nei;
                
                neigh_grid_begin= bins[nei_gird_index].begin();
                neigh_grid_end  = bins[nei_gird_index].end();
                
                for(cur = grid_begin; cur!= grid_end; ++cur){
                    for(neighbor = neigh_grid_begin; neighbor != neigh_grid_end; ++neighbor){
                        apply_force(**cur, **neighbor);
                    }
                }
            }
        }
	}
	// Move Particles and update each particle's bin
#pragma omp for schedule(dynamic, CHUNKSIZE)
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}