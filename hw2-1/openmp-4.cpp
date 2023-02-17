#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <algorithm>

using namespace std;
#define BINSIZE (cutoff * 2.1)

int bin_dim;
vector< vector<particle_t*> > bins;
vector<omp_lock_t> bin_locks;
int directions[9][2] = {{-1, 1}, {-1, 0}, {-1, -1}, {0, 1}, {0, 0}, {0, -1}, {1, 1}, {1, 0}, {1, -1}};

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

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method

    // original bin
    int bin_x = floor(p.x / BINSIZE);
    int bin_y = floor(p.y / BINSIZE);
    const int ori_bin = bin_x * bin_dim + bin_y;

    p.x += (p.vx += p.ax * dt) * dt;
    p.y += (p.vy += p.ay * dt) * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    // new bin
    int new_x = floor(p.x / BINSIZE);
    int new_y = floor(p.y / BINSIZE);
    const int new_bin = new_x * bin_dim + new_y;

    if (bin_x == new_x and bin_y == new_y){
        return;
    }

    // Delete from the original bin
    omp_set_lock(&bin_locks[ori_bin]);
    vector<particle_t*> &bin = bins[ori_bin];
    for(int i = 0; i < bin.size(); ++i){
        if(bin[i] == &p){
            bin.erase(bin.begin() + i);
            break;
        }
    }
    omp_unset_lock(&bin_locks[ori_bin]);

    // Move to the new bin
    omp_set_lock(&bin_locks[new_bin]);
    bins[new_bin].push_back(&p);
    omp_unset_lock(&bin_locks[new_bin]);

    return;
}
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    bin_dim = floor(size / BINSIZE) + 1;

    bins = vector<vector<particle_t*>>(bin_dim * bin_dim);
    bin_locks = vector<omp_lock_t>(bin_dim * bin_dim);

    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        int bin_x = floor(parts[i].x / BINSIZE); 
        int bin_y = floor(parts[i].y / BINSIZE);
        bins[bin_x * bin_dim+ bin_y].push_back(&parts[i]);
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces
#pragma omp for
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }

#pragma omp for collapse(2)
    for(int i = 0; i < bin_dim; ++i){
        for(int j = 0; j < bin_dim; ++j){
            auto &bin = bins[i * bin_dim + j];
            const int bin_size = bin.size();

            for(int d = 0; d < 9; ++d){
                int neighbor_binx = i + directions[d][0];
                int neighbor_biny = j + directions[d][1];
                if(neighbor_binx < 0 or neighbor_binx >= bin_dim or neighbor_biny < 0 or neighbor_biny >= bin_dim)
                    continue;
                auto &neighbor_bin = bins[neighbor_binx * bin_dim + neighbor_biny];
                const int neighbor_size = neighbor_bin.size();
                for(int l = 0; l < bin_size; ++l){
                    particle_t *cur = bin[l];
                    for(int k = 0; k < neighbor_size; ++k){
                        particle_t *neighbor = neighbor_bin[k];
                        apply_force(*cur, *neighbor);
                    }
                }
            }
        }
    }

#pragma omp for
    for(int i = 0; i < num_parts; ++i){
        move(parts[i], size);
    }
}

