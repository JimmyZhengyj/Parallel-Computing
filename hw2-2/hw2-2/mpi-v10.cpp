#include "common.h"
#include <mpi.h>

#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <assert.h>
#include <cmath>

#define BINSIZE (cutoff * 2)

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
std::vector<std::vector<particle_t>> bins;

// rank_grids_send[rank] = the grids I need to send to you
// NOTE: I think the order is vital otherwise Isend, Irecv might be wrong
std::map<int, std::vector<int>> rank_grids_send;
std::map<int, std::vector<int>> rank_grids_recv;

using Len = int;

std::map<int, std::array<MPI_Request, 2>> sendreqs;
std::map<int, std::array<MPI_Request, 2>> recvreqs;

using Rank = int;
// send_parts[target_rank] = the particles I want to send to you
std::map<Rank, Len> send_lens;
std::map<Rank, std::vector<particle_t>> send_parts;
// recv_parts[src_rank] = the particles I will place upon once I receive it
std::map<int, std::vector<particle_t>> recv_parts;

std::vector<int> num_send_parts;
std::vector<int> num_recv_parts;
std::vector<int> send_displ;
std::vector<int> recv_displ;
std::vector<particle_t> local_parts_send;
std::vector<particle_t> local_parts_recv;

// dim x dim
int dim;
int dim_square;
int q, r;
// bins information
int local_offset;
int local_start;
int num_bins;
int num_bins_w_neighbors;

int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

int get_global_bin_idx(double x, double y){
    return floor(x / BINSIZE) * dim + floor(y / BINSIZE);
}

constexpr int get_bin_i(double x){
    return floor(x / BINSIZE);
}
constexpr int get_bin_j(double y){
    return floor(y / BINSIZE);
}

int get_rank_from_bin_idx(int bidx){
    // check if it belongs to the first r processes
    return (bidx < r * (q + 1))? (bidx / (q+1)):((bidx - r * (q + 1)) / q + r);
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

void apply_force_left(particle_t& particle, particle_t& neighbor) {
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

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method

    // Update vx, vy and directly use there return value to update x,y
    p.x += (p.vx += p.ax * dt) * dt;
    p.y += (p.vy += p.ay * dt) * dt;

    // reset (to denote that this particle has finished)
    p.ax = p.ay = 0;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
    return;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    dim_square = dim * dim;

    // Each process will at least do `q` number of works
    q = dim_square / num_procs;
    /* if(rank == 0){ */
    /*     std::cout << "q: " << q << std::endl; */
    /*     std::cout << "dim_square: " << dim_square << std::endl; */
    /* } */
    // The first r processes will do 1 additional work
    r = dim_square % num_procs;

    if(rank < r){
        local_start = rank * (q+1);
    }else{
        local_start = r * (q+1) + (rank - r) * q;
    }

    // # of works(bins) to do
    num_bins = (rank < r)? (q+1):(q);

    // NOTE: sorted
    std::unordered_map<int, std::set<int>> temp_rank_grids_send;
    std::unordered_map<int, std::set<int>> temp_rank_grids_recv;

    // We need to for each process, which grid we need to send to him
    for(int i = 0; i < num_bins; ++i){
        int bi = (local_start + i) / dim;
        int bj = (local_start + i) % dim;
        for(int d = 0; d < 8; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nei_rank = get_rank_from_bin_idx(ni * dim + nj);
            // If it is still in this process, skip
            if(nei_rank == rank)
                continue;
            // I need to send this grid to this process
            // NOTE: need to avoid duplicate grid indices
            temp_rank_grids_send[nei_rank].insert(local_start + i);
            // I expect to receive this grid from this process
            temp_rank_grids_recv[nei_rank].insert(ni * dim + nj);
        }
    }

    // Initialize reqs
    for(auto &kv: temp_rank_grids_send){
        int target_src_rank = kv.first;
        auto &grid = kv.second;
        for(int i: grid){
            rank_grids_send[target_src_rank].push_back(i);
        }

        auto &recv_grid = temp_rank_grids_recv[target_src_rank];
        for(int bidx: recv_grid){
            rank_grids_recv[target_src_rank].push_back(bidx);
        }

        // pre-initialize to prevent the tree change its underlying memory
        // NOTE: I don't think this is actually a problem 
        send_lens[target_src_rank] = 0;

        // pre-initalize
        sendreqs[target_src_rank];
        recvreqs[target_src_rank];

        // Pre-allocate a large space
        send_parts[target_src_rank].reserve(num_parts / num_procs);
        recv_parts[target_src_rank].reserve(num_parts / num_procs);
    }

    int first_bi = local_start / dim;
    int first_bj = local_start % dim;

    int last_bi = (local_start + num_bins - 1) / dim;
    int last_bj = (local_start + num_bins - 1) % dim;

    // the last (bi, bj)'s fartherest neighbor grid
    // [up_farthest_bidx, down_farthest_bidx]
    // but we will only modify [local_start, local_start + num_bins] particles
    int up_farthest_bidx = MAX((first_bi - 1) * dim + (first_bj - 1), 0);
    local_offset = up_farthest_bidx;
    int down_farthest_bidx = MIN((last_bi + 1) * dim + (last_bj + 1), dim_square-1);
    // we are also responsible for computing forces for some of grids in this range
    num_bins_w_neighbors = down_farthest_bidx - up_farthest_bidx + 1;
    // Resize it to include the neighbors on the bottom or right
    bins.resize(num_bins_w_neighbors);

    // TODO: Put particles into bins
    // Can this be parallelized?
    for(int i = 0; i < num_parts; ++i){
        int bidx = get_global_bin_idx(parts[i].x, parts[i].y);
        if(local_start <= bidx and bidx < local_start + num_bins){
            bins[bidx - local_offset].push_back(parts[i]);
        }
    }
    // Make sure all the process collects the particles it need to compute forces
    MPI_Barrier(MPI_COMM_WORLD);

    num_send_parts.resize(num_procs);
    num_recv_parts.resize(num_procs);
    send_displ.resize(num_procs);
    recv_displ.resize(num_procs);
    // at most `num_parts` (for root process)
    local_parts_send.reserve(num_parts);
    local_parts_recv.reserve(num_parts);
}


// void send_grid(int target_rank,
//         int &n,
//         std::vector<particle_t> &grid,
//         int bidx,
//         std::array<MPI_Request, 2> &reqs
//         ){

//     // NOTE: n must be reference because n need to visible after this function ends
//     MPI_Isend(&n, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &reqs[0]);
//     MPI_Isend(grid.data(), n, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
// }

void send_particles(int target_rank, int &n,
        std::vector<particle_t> &particles,
        std::array<MPI_Request, 2> &reqs){

    MPI_Isend(&n, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(particles.data(), n, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void recv_particles(int src_rank,
        std::vector<particle_t> &particles,
        std::array<MPI_Request, 2> &reqs){
    particles.clear();
    int n;
    MPI_Recv(&n, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    particles.resize(n);
    MPI_Irecv(particles.data(), n, PARTICLE, src_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

int sum(std::vector<int> &v){
    int s = 0;
    for(int e: v){
        s += e;
    }
    return s;
}

void exclusive_psum(std::vector<int> &v, std::vector<int> &p){
    int s = 0;
    int n = v.size();
    for(int i = 0; i < n; ++i){
        p[i] = s;
        s += v[i];
    }
}

void clear_grids_not_owned_by_me(){
    for(int i = local_offset; i < local_start; ++i){
        bins[i - local_offset].clear();
    }
    for(int i = local_start + num_bins; i < local_offset + num_bins_w_neighbors; ++i){
        bins[i - local_offset].clear();
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // 1. Each process collect and send the neighbor's (x, y) information

    // Send
    for(auto &kv: rank_grids_send){
        int target_rank = kv.first;
        auto &grid_indices = kv.second;
        int n = grid_indices.size();

        auto &reqs = sendreqs[target_rank];

        // Gather all particles to reduce communication overhead
        auto &parts_to_send = send_parts[target_rank];
        int &len = send_lens[target_rank];

        len = 0; // reset to zero
        parts_to_send.clear(); // clear all the particles
        for(int i = 0; i < n; ++i){
            int bidx = grid_indices[i];
            /* assert(get_rank_from_bin_idx(bidx) == rank); */
            auto &grid = bins[bidx - local_offset];
            int grid_n = grid.size();
            len += grid_n;
            for(int j = 0; j < grid_n; ++j)
                parts_to_send.push_back(grid[j]);
        }
        // Send all the particles at once!!!
        send_particles(target_rank, len, parts_to_send, reqs);
    }

    // Receive
    for(auto &kv: rank_grids_recv){
        int src_rank = kv.first;

        auto &reqs = recvreqs[src_rank];
        auto &parts_to_recv = recv_parts[src_rank];

        // Recv all the particles at once!!!
        recv_particles(src_rank, parts_to_recv, reqs);
    }

    // Wait to make sure you actually send it out
    for(auto &kv: sendreqs){
        auto &req = kv.second;

        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }
    // clear neighbor grids because we will put the new particles inside later
    clear_grids_not_owned_by_me();
    // Wait to make sure you actually receive it
    for(auto &kv: recvreqs){
        int src_rank = kv.first;
        auto &req = kv.second;
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        // Put the received particles to the correct bins
        auto &parts_to_recv = recv_parts[src_rank];
        int n = parts_to_recv.size();
        for(int i = 0; i < n; ++i){
            particle_t &p = parts_to_recv[i];
            int bidx = get_global_bin_idx(p.x, p.y);
            /* assert(get_rank_from_bin_idx(bidx) != rank); */
            /* assert(0 <= bidx and bidx - local_offset < num_bins_w_neighbors); */
            bins[bidx - local_offset].push_back(p);
        }
    }

    // Clear the accelerations
    for(int i = 0; i < num_bins; ++i){
        auto &grid = bins[i + (local_start - local_offset)];
        for(particle_t &p: grid){
            p.ax = p.ay = 0;
        }
    }

    // Compute bidirectional forces
    for(int i = 0; i < num_bins; ++i){
        auto &grid = bins[i + (local_start - local_offset)];
        int bi = (local_start + i) / dim;
        int bj = (local_start + i) % dim;
        // Compute the forces within the grid
        int grid_n = grid.size();
        for(int j = 0; j < grid_n; ++j){
            for(int k = j+1; k < grid_n; ++k){
                apply_force_bidir(grid[j], grid[k]);
            }
        }
        // compute forces within this grid
        for(int d = 0; d < 8; ++d){
            int ni = bi + dir[d][0];
            int nj = bj + dir[d][1];
            if(ni < 0 or ni >= dim or nj < 0 or nj >= dim)
                continue;
            int nidx = ni * dim + nj;
            auto &nei_grid = bins[nidx - local_offset];

            for(particle_t &p1: grid){
                for(particle_t &p2: nei_grid){
                    // only compute the forces for itself
                    apply_force_left(p1, p2);
                }
            }
        }
    }

    // Move and collect those particles that need to be sent to other processes
    // need_dist[target_rank] = a list of particles I need to send to target_rank
    std::map<int, std::vector<particle_t>> need_redist;
    std::vector<particle_t> parts_moved_but_still_mine;
    for(int i = 0; i < num_bins; ++i){
        auto &grid = bins[i + (local_start - local_offset)];
        auto it = grid.begin();
        int grid_n = grid.size();
        // Loop from the back because we are going to delete the particle
        // as we loop over it
        for(int j = grid_n-1; j >= 0; --j){
            // if this particle has been processed (because of bin redistributing)
            // continue
            particle_t &p = grid[j];
            /* assert(get_global_bin_idx(p.x, p.y) == local_start + i); */
            // update (x, y, vx, vy) from (ax, ay) information
            // and reset (ax, ay)
            move(p, size);
            //
            int new_bidx = get_global_bin_idx(p.x, p.y);
            if(new_bidx == local_start + i){
                // no move
                continue;
            }else if(local_start <= new_bidx and new_bidx < local_start + num_bins){
                // move, but still in this process (i.e. rank)
                parts_moved_but_still_mine.push_back(p);
                // delete this particle (that's why loop from the back)
                grid.erase(it + j);
            }else{
                // need to send this particle to the other process (i.e. rank)
                int target_rank = get_rank_from_bin_idx(new_bidx);
                need_redist[target_rank].push_back(p);
                // delete this particle
                grid.erase(it + j);
            }
        }
    }
    // Put into the right bin
    for(particle_t &p: parts_moved_but_still_mine){
        int bidx = get_global_bin_idx(p.x, p.y);
        bins[bidx - local_offset].push_back(p);
    }

    // --------------------------------------------------------
    // Particle redistribution
    int send_total = 0;
    for(int i = 0; i < num_procs; ++i)
        num_send_parts[i] = 0;
    for(auto &kv: need_redist){
        int target_rank = kv.first;
        auto &ps = kv.second;
        num_send_parts[target_rank] += ps.size();
        send_total += ps.size();
    }
    // num_recv_parts[i] == # of particles process i is gonna send me

    // Send: how many particles am I gonna send to each processor
    // Recv: how many particles will I expect to receive from each processor
    MPI_Alltoall(&num_send_parts[0], 1, MPI_INT,
                 &num_recv_parts[0], 1, MPI_INT,
                 MPI_COMM_WORLD);

    int recv_total = sum(num_recv_parts);

    local_parts_recv.resize(recv_total);

    // flatten
    local_parts_send.clear();
    for(auto &kv: need_redist){
        for(particle_t &t: kv.second){
            local_parts_send.push_back(t);
        }
    }
    /* assert(local_parts_send.size() == send_total); */
    // compute the displacement
    exclusive_psum(num_send_parts, send_displ);
    exclusive_psum(num_recv_parts, recv_displ);

    // Send and recv the particles
    // implicit barrier
    MPI_Alltoallv(&local_parts_send[0], &num_send_parts[0], &send_displ[0], PARTICLE,
                &local_parts_recv[0], &num_recv_parts[0], &recv_displ[0], PARTICLE,
                MPI_COMM_WORLD);

    // Put the local_parts_recv into the right bins
    for(particle_t &p: local_parts_recv){
        int bidx = get_global_bin_idx(p.x, p.y);
        /* assert(local_start <= bidx and bidx < local_start + num_bins); */
        bins[bidx - local_offset].push_back(p);
    }
    return;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Each process tell the root process:
    // how many particles will I send you
    // and then gather the particles the root needs
    int num_parts_to_send = 0;

    local_parts_send.clear();

    // Flatten
    for(int i = 0; i < num_bins; ++i){
        int n = bins[i + (local_start - local_offset)].size();
        num_parts_to_send += n;
        for(int j = 0; j < n; ++j){
            local_parts_send.push_back(bins[i + (local_start - local_offset)][j]);
        }
    }

    if(rank == 0){
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, num_recv_parts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // count how many elements will I receive
    if(rank == 0){
        // the root expects to receive `total` number of particles
        int total = sum(num_recv_parts);
        local_parts_recv.resize(total);
        /* if(total != num_parts){ */
        /*     std::cout << "total: " << total << std::endl; */
        /*     assert(total == num_parts); */
        /* } */

        // Compute the displacement
        exclusive_psum(num_recv_parts, recv_displ);

        MPI_Gatherv(local_parts_send.data(), local_parts_send.size(),
            PARTICLE, local_parts_recv.data(),
            num_recv_parts.data(), recv_displ.data(),
            PARTICLE, 0, MPI_COMM_WORLD);

        // put pack all particles
        for(particle_t &p: local_parts_recv){
            parts[p.id-1] = p;
        }
    }else{
        MPI_Gatherv(local_parts_send.data(), local_parts_send.size(),
            PARTICLE, nullptr,
            nullptr, nullptr,
            PARTICLE, 0, MPI_COMM_WORLD);
    }
}
