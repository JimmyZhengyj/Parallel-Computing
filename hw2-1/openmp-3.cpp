#include "common.h"
#include <cmath>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;
#define BINSIZE (cutoff *1)

int bin_dim;
int directions[9][2] = {{-1, 1}, {-1, 0}, {-1, -1}, {0, 1}, {0, 0}, {0, -1}, {1, 1}, {1, 0}, {1, -1}};
vector< unordered_set<particle_t*> > bins;

// Apply the force from neighbor to particle
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
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
  
    int bin_x = floor(p.x / BINSIZE);
    int bin_y = floor(p.y / BINSIZE);
    int cur_bin = bin_x * bin_dim + bin_y;

  
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;


    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    int new_x = floor(p.x / BINSIZE);
    int new_y = floor(p.y / BINSIZE);
    
    int new_bin = new_x * bin_dim + new_y;

    if (cur_bin != new_bin){ 
        return;
    }

#pragma omp critical
    {
        bins[cur_bin].erase(&p);
        bins[new_bin].insert(&p);
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    omp_set_num_threads(20);

    bin_dim = floor(size / BINSIZE) + 1;
    bins = vector< unordered_set<particle_t*> >(bin_dim * bin_dim);
    
#pragma omp for schedule(dynamic, CHUNKSIZE)
    for(int i = 0; i < num_parts; ++i){
        int bin_x = floor(parts[i].x / BINSIZE); 
        int bin_y = floor(parts[i].y / BINSIZE);
        bins[bin_x * bin_dim+ bin_y].insert(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
	
        int bin_x = floor(parts[i].x / BINSIZE);
        int bin_y = floor(parts[i].y / BINSIZE);
        
        for (int d = 0; d < 9; ++d){
	 // std::cout<<d<<' ';
            int neigh_binx = bin_x + directions[d][0];
            int neigh_biny = bin_y + directions[d][1];

            if (neigh_binx < 0 or neigh_binx >= bin_dim or neigh_biny < 0 or neigh_biny >= bin_dim){
                continue;
            }

	    
            unordered_set<particle_t*> neighs = bins[neigh_binx * bin_dim + neigh_biny];
	    
            for (auto neigh=neighs.begin(); neigh != neighs.end(); ++neigh){
                apply_force (parts[i], **neigh);
            }
        }

    }

    // Move Particles
#pragma omp for schedule(dynamic, CHUNKSIZE)
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}