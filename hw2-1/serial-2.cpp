#include "common.h"
#include <cmath>

#include <vector>
#include <algorithm>
#include <iostream>

// 83 for 0.1M
/* #define BINSIZE (0.01 + 0.001) */
// #define BINSIZE (cutoff * 1.1) 
// 79 for 0.1M
// #define BINSIZE (cutoff * 1.2) 
// 76 for 0.1M
// #define BINSIZE (cutoff * 1.3)
// 73 for 0.1M
// #define BINSIZE (cutoff * 1.4)
// 71 for 0.1M
// #define BINSIZE (cutoff * 1.5)
// 69 for 0.1M (714 for 1M)
// #define BINSIZE (cutoff * 1.6)
// 68 for 0.1M
// #define BINSIZE (cutoff * 1.7)
// 66 for 0.1M
// #define BINSIZE (cutoff * 1.8)
// 64 for 0.1M
// #define BINSIZE (cutoff * 1.9) 
// 63.7 for 0.1M
// #define BINSIZE (cutoff * 2.0)
// 58 for 0.1M (611 for 1M)
#define BINSIZE (cutoff * 2.1)
// 59 for 0.1M
// #define BINSIZE (cutoff * 2.2) 
// #define BINSIZE (cutoff * 2.3)

#define EXPERIMENT 0
//0: all optimize
//1: all but bi-force
//2: wo bi-force and reserved
//3: wo locality

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}
// bin[i][j] = the particles
std::vector<std::vector<particle_t*>> bins;
int griddim;

// Ensure loop over row by row
#if EXPERIMENT == 3
int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
#else 
int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
#endif


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
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

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
    double x_ori = p.x;
    double y_ori = p.y;

    // Update vx, vy and directly use there return value to update x,y
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

    // no change, return
    // Probabilistically, it will almost never happen,
    // so comment it
    /* if(p.x == x_ori and p.y == y_ori) */
    /*     return; */
    // no change in bin
    if(bi(x_ori) == bi(p.x) and bj(y_ori) == bj(p.y)) return;

    // the coordinate changes, update this particle to a correct bin
    std::vector<particle_t*> &grid = bins[bi(x_ori) * griddim + bj(y_ori)];
    int grid_n = grid.size();
    // delete the particle from the original grid
    // NOTE: grid_n should be constant if the density for each grid is a constant
    for(int i = 0; i < grid_n; ++i){
        if(grid[i] == &p){
            grid.erase(grid.begin() + i);
            break;
        }
    }
    // insert the particle to the correct grid
    bins[bi(p.x) * griddim + bj(p.y)].push_back(&p);
    return;
}


#if EXPERIMENT == 0
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            // reserve the 1.2 * # of expected particles
            bins[i * griddim + j].reserve(space);
        }
    }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }

    /* std::cout << "Running experiment: " << EXPERIMENT << "\n"; */
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the acceleration
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }
    // Loop over each grid (better locality)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            auto &grid = bins[i * griddim + j];
            // Loop over each particle in this grid
            const int grid_n = grid.size();
            // For each neighbor grid,
            // apply the force to the particles inside this grid
            // NOTE: the inner loop must start from l+1,
            // otherwise, we will compute the force twice for particles
            // in the same grid
            for(int l = 0; l < grid_n; l++){
                particle_t *cur = grid[l];
                for(int k = l+1; k < grid_n; ++k){
                    particle_t *neighbor = grid[k];
                    // apply_force(*cur, *neighbor); 
                    // apply_force(*neighbor, *cur); 
                    apply_force_bidir(*cur, *neighbor);
                }
            }
            for(int d = 5; d < 9; ++d){
                int bi_nei = i + dir[d][0];
                int bj_nei = j + dir[d][1];
                // out of bound
                if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                    continue;
                // Apply forces to all the particles inside this grid
                auto &neighbor_grid = bins[bi_nei * griddim + bj_nei];
                const int neighbor_grid_n = neighbor_grid.size();
                for(int l = 0; l < grid_n; ++l){
                    particle_t *cur = grid[l];
                    for(int k = 0; k < neighbor_grid_n; ++k){
                        particle_t *neighbor = neighbor_grid[k];
                        // apply_force(*cur, *neighbor);
                        // apply_force(*neighbor, *cur);
                        apply_force_bidir(*cur, *neighbor);
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

#elif EXPERIMENT == 1
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            // reserve the 1.2 * # of expected particles
            bins[i * griddim + j].reserve(space);
        }
    }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }

    /* std::cout << "Running experiment: " << EXPERIMENT << "\n"; */
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the acceleration
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }
    // Loop over each grid (better locality)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            auto &grid = bins[i * griddim + j];
            // Loop over each particle in this grid
            const int grid_n = grid.size();
            // For each neighbor grid,
            // apply the force to the particles inside this grid
            // NOTE: the inner loop must start from l+1,
            // otherwise, we will compute the force twice for particles
            // in the same grid
            for(int l = 0; l < grid_n; l++){
                particle_t *cur = grid[l];
                for(int k = l+1; k < grid_n; ++k){
                    particle_t *neighbor = grid[k];
                    apply_force(*cur, *neighbor); 
                    apply_force(*neighbor, *cur); 
                    // apply_force_bidir(*cur, *neighbor);
                }
            }
            for(int d = 5; d < 9; ++d){
                int bi_nei = i + dir[d][0];
                int bj_nei = j + dir[d][1];
                // out of bound
                if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                    continue;
                // Apply forces to all the particles inside this grid
                auto &neighbor_grid = bins[bi_nei * griddim + bj_nei];
                const int neighbor_grid_n = neighbor_grid.size();
                for(int l = 0; l < grid_n; ++l){
                    particle_t *cur = grid[l];
                    for(int k = 0; k < neighbor_grid_n; ++k){
                        particle_t *neighbor = neighbor_grid[k];
                        apply_force(*cur, *neighbor);
                        apply_force(*neighbor, *cur);
                        // apply_force_bidir(*cur, *neighbor);
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

#elif EXPERIMENT == 2
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    // for(int i = 0; i < griddim; ++i){
    //     for(int j = 0; j < griddim; ++j){
    //         // reserve the 1.2 * # of expected particles
    //         bins[i * griddim + j].reserve(space);
    //     }
    // }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }

    /* std::cout << "Running experiment: " << EXPERIMENT << "\n"; */
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the acceleration
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }
    // Loop over each grid (better locality)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            auto &grid = bins[i * griddim + j];
            // Loop over each particle in this grid
            const int grid_n = grid.size();
            // For each neighbor grid,
            // apply the force to the particles inside this grid
            // NOTE: the inner loop must start from l+1,
            // otherwise, we will compute the force twice for particles
            // in the same grid
            for(int l = 0; l < grid_n; l++){
                particle_t *cur = grid[l];
                for(int k = l+1; k < grid_n; ++k){
                    particle_t *neighbor = grid[k];
                    apply_force(*cur, *neighbor); 
                    apply_force(*neighbor, *cur); 
                    // apply_force_bidir(*cur, *neighbor);
                }
            }
            for(int d = 5; d < 9; ++d){
                int bi_nei = i + dir[d][0];
                int bj_nei = j + dir[d][1];
                // out of bound
                if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                    continue;
                // Apply forces to all the particles inside this grid
                auto &neighbor_grid = bins[bi_nei * griddim + bj_nei];
                const int neighbor_grid_n = neighbor_grid.size();
                for(int l = 0; l < grid_n; ++l){
                    particle_t *cur = grid[l];
                    for(int k = 0; k < neighbor_grid_n; ++k){
                        particle_t *neighbor = neighbor_grid[k];
                        apply_force(*cur, *neighbor);
                        apply_force(*neighbor, *cur);
                        // apply_force_bidir(*cur, *neighbor);
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
#elif EXPERIMENT == 3

void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    // for(int i = 0; i < griddim; ++i){
    //     for(int j = 0; j < griddim; ++j){
    //         // reserve the 1.2 * # of expected particles
    //         bins[i * griddim + j].reserve(space);
    //     }
    // }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }

    /* std::cout << "Running experiment: " << EXPERIMENT << "\n"; */
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the acceleration
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }
    // Loop over each grid (better locality)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            auto &grid = bins[i * griddim + j];
            // Loop over each particle in this grid
            const int grid_n = grid.size();
            // For each neighbor grid,
            // apply the force to the particles inside this grid
            // NOTE: the inner loop must start from l+1,
            // otherwise, we will compute the force twice for particles
            // in the same grid
            for(int l = 0; l < grid_n; l++){
                particle_t *cur = grid[l];
                for(int k = l+1; k < grid_n; ++k){
                    particle_t *neighbor = grid[k];
                    apply_force(*cur, *neighbor); 
                    apply_force(*neighbor, *cur);
                    // apply_force_bidir(*cur, *neighbor);
                }
            }
            for(int d = 0; d < 8; ++d){
                int bi_nei = i + dir[d][0];
                int bj_nei = j + dir[d][1];
                // out of bound
                if(bi_nei < 0 or bi_nei >= griddim or bj_nei < 0 or bj_nei >= griddim)
                    continue;
                // Apply forces to all the particles inside this grid
                auto &neighbor_grid = bins[bi_nei * griddim + bj_nei];
                const int neighbor_grid_n = neighbor_grid.size();
                for(int l = 0; l < grid_n; ++l){
                    particle_t *cur = grid[l];
                    for(int k = 0; k < neighbor_grid_n; ++k){
                        particle_t *neighbor = neighbor_grid[k];
                        apply_force(*cur, *neighbor);
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

#endif
