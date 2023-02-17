#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>

/* #include <iostream> */
#include <algorithm>

#define BINSIZE (cutoff * 2.1)
#define NUMTHREADS 20

constexpr int bi(double x){
    return floor(x / BINSIZE);
}
constexpr int bj(double y){
    return floor(y / BINSIZE);
}

struct particle_pos{
    double x, y;
};

struct particle_t_w_addr{
    particle_t *addr; // for write back
    double x;  // Position X
    double y;  // Position Y
    double vx;
    double vy;
    double ax; // Acceleration X
    double ay; // Acceleration Y
};

int griddim;
std::vector<std::vector<particle_t*>> bins;
std::vector<std::vector<particle_t_w_addr>> write_bins;
std::vector<omp_lock_t> bin_locks;

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

    // no change in bin
    if(bi(x_ori) == bi(p.x) and bj(y_ori) == bj(p.y)) return;

    const int from = bi(x_ori) * griddim + bj(y_ori);
    const int to = bi(p.x) * griddim + bj(p.y);

    // Lock the `from` grid
    omp_set_lock(&bin_locks[from]);
    // the coordinate changes, update this particle to a correct bin
    std::vector<particle_t*> &grid = bins[from];
    int grid_n = grid.size();
    // delete the particle from the original grid
    // NOTE: grid_n should be constant if the density for each grid is a constant
    for(int i = 0; i < grid_n; ++i){
        if(grid[i] == &p){
            grid.erase(grid.begin() + i);
            break;
        }
    }
    omp_unset_lock(&bin_locks[from]);
    omp_set_lock(&bin_locks[to]);
    bins[to].push_back(&p);
    omp_unset_lock(&bin_locks[to]);
    return;
}
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    griddim = floor(size / BINSIZE) + 1;

    bins = std::vector<std::vector<particle_t*>>(griddim * griddim);
    bin_locks = std::vector<omp_lock_t>(griddim * griddim);

    const int space = ceil(1.2 * BINSIZE * BINSIZE * 1. / density);
    // Pre-reserve the memory at once
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            // reserve the 1.2 * # of expected particles
            int idx = i * griddim + j;
            bins[idx].reserve(space);

            //init bin_locks
            omp_init_lock(&bin_locks[idx]);
        }
    }
    // Put particles into the bins
    for(int i = 0; i < num_parts; ++i){
        double x = parts[i].x;
        double y = parts[i].y;

        bins[bi(x) * griddim + bj(y)].push_back(&parts[i]);
    }
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
    static int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
#pragma omp for
    for(int i = 0; i < num_parts; ++i){
        parts[i].ax = parts[i].ay = 0;
    }

#pragma omp for collapse(2)
    for(int i = 0; i < griddim; ++i){
        for(int j = 0; j < griddim; ++j){
            auto &grid = bins[i * griddim + j];
            const int grid_n = grid.size();

            for(int l = 0; l < grid_n; ++l){
                particle_t *cur = grid[l];
                for(int k = l+1; k < grid_n; ++k){
                    particle_t *neighbor = grid[k];
                    apply_force(*cur, *neighbor);
                }
            }

            for(int d = 5; d < 9; ++d){
                int ni = i + dir[d][0];
                int nj = j + dir[d][1];
                if(ni < 0 or ni >= griddim or nj < 0 or nj >= griddim)
                    continue;
                auto &neighbor_grid = bins[ni * griddim + nj];
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
#pragma omp for
    for(int i = 0; i < num_parts; ++i){
        move(parts[i], size);
    }
}

