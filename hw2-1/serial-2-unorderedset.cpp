#include "common.h"
#include <cmath>
// #include <bits/stdc++.h>
#include <iterator>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

// 83 for 0.1M
/* #define BINSIZE (0.01 + 0.001) */
// 79 for 0.1M
/* #define BINSIZE (cutoff * 1.2) */
// 76 for 0.1M
/* #define BINSIZE (cutoff * 1.3) */
// 73 for 0.1M
/* #define BINSIZE (cutoff * 1.4) */
// 71 for 0.1M
/* #define BINSIZE (cutoff * 1.5) */
// 69 for 0.1M (714 for 1M)
/* #define BINSIZE (cutoff * 1.6) */
// 68 for 0.1M
/* #define BINSIZE (cutoff * 1.7) */
// 66 for 0.1M
/* #define BINSIZE (cutoff * 1.8) */
// 64 for 0.1M
/* #define BINSIZE (cutoff * 1.9) */
// 63.7 for 0.1M
/* #define BINSIZE (cutoff * 2.0) */
// 58 for 0.1M (611 for 1M)
// #define BINSIZE (cutoff * 2.1)
// 59 for 0.1M
/* #define BINSIZE (cutoff * 2.2) */
#define BINSIZE (cutoff * 2.3)

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}
// bin[i][j] = the particles
vector<unordered_set<particle_t* >> bins;
int griddim;

// Ensure loop over row by row
int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

// Apply the force from neighbor to particle
void apply_force_bidir(particle_t& particle, particle_t& neighbor) {
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

    particle.ax += coef * dx;
    particle.ay += coef * dy;

    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    int orig_bin = bi(p.x) * griddim + bj(p.y);

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

    int new_bin = bi(p.x) * griddim + bj(p.y);

    if(orig_bin == new_bin) return;
    //update the bin
    
    bins[orig_bin].erase(&p);
    bins[new_bin].insert(&p);
 
    return;
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    bins = vector<unordered_set<particle_t*>>(griddim * griddim);
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;
        bins[bi(x) * griddim + bj(y)].insert(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the acceleration
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }
    // Loop over each grid (better locality)
    unordered_set<particle_t* > :: iterator cur, neighbor, grid_begin, grid_end, neigh_grid_begin, neigh_grid_end;

    for(int i = 0; i < griddim; ++i){
        int i_offset = i * griddim;
        for(int j = 0; j < griddim; ++j){
            int bin_index = i_offset + j;
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
                    apply_force_bidir(**cur, **neighbor);
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
                        apply_force_bidir(**cur, **neighbor);
                    }
                }
            }
        }
    }

    // Move Particles and update each particle's bin
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
