#include "common.h"
#include <mpi.h>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

#define BINSIZE (cutoff * 1)

// Global Variables
int bin_dim;
int bin_dim_x;
const int directions[9][2] = {{-1, 1}, {-1, 0}, {-1, -1}, {0, 1}, {0, 0}, {0, -1}, {1, 1}, {1, 0}, {1, -1}};
vector<vector<particle_t>> bins;
int ghost_factor;
vector<particle_t> incoming_particles, out_down_particles, out_up_particles;
int subdomain_start; // index where the subdomain bin row starts
int subdomain_size;  // number of bin rows in subdomain


void decompose_domain(int world_rank, int world_size) {
/** Distribute bins amongst the processors in rows (eventually change to rectangular).
    Output changes subdomain_start and subdomain_size.
    Inputs:
      world_rank: rank of processors in MPI_COMM_WORLD
      world_size: number of processors in MPI_COMM_WORLD
*/
  // special case --> add case later (restrict comms to just the ones needed)

  if (world_size > bin_dim) {
    // std::cout << "working" <<endl;
    MPI_Abort(MPI_COMM_WORLD, 1); 
  }

  subdomain_start = bin_dim / world_size * world_rank;
  subdomain_size = bin_dim / world_size;
  // give remainder of rows to last process
  if (world_rank == world_size - 1) {
    subdomain_size += bin_dim % world_size;
  }
}

void bin_ind(particle_t& part, int subdomain_start, int subdomain_size,
	     int rank, int numProcs, int& bi){
  int bin_x = floor(part.x / BINSIZE);
  int bin_y = floor(part.y / BINSIZE);
  if ((bin_x >= subdomain_start) && (bin_x < subdomain_start + subdomain_size)){
    bi = (bin_x-subdomain_start) * bin_dim + bin_y;
    return;
  }
  // ghost atoms
  else if(bin_x == subdomain_start - 1){
    bi = bin_y;
    return;
  }
  else if(bin_x == subdomain_start + subdomain_size){
    bi = (subdomain_size+ghost_factor-1)*bin_dim + bin_y;
    return;
  }
  else if (bin_x > subdomain_start+subdomain_size){
    bi = -2;
    return;
  }
  else if (bin_x < subdomain_start-1){
    bi = -3;
    return;
  }
  else return;// can return something that tells whether it goes above or below
}

void initialize_bins(particle_t* parts, int num_parts,
		     int subdomain_start, int subdomain_size,
		     int rank, int size) {

  ghost_factor = 2; // how many rows to add for ghost bins
  bin_dim_x = subdomain_size+ghost_factor;

  bins = vector<vector<particle_t>>(bin_dim * bin_dim_x);
  // bins = vector<vector<particle_t*>>(num_parts);
  for (int i = 0; i < num_parts; i++) {
    int bi = -1; // index of bin to put particle in
    bin_ind(parts[i], subdomain_start, subdomain_size, rank, size, bi);
    // std::cout << bi<<" \n";
    if (bi >= 0){
      bins[bi].push_back(parts[i]);
    }
  }
}

void rebin_particles(vector<particle_t> incoming_particles, int rank, int num_procs){
  // std::cout << "Begin Rebin Particle: \n";
  int vecsize = incoming_particles.size();
  for (int i = 0; i < vecsize; i++){
    int bi = -1;
    // particle_t& temp = incoming_particles[i];
    bin_ind(incoming_particles[i], subdomain_start, subdomain_size,rank, num_procs, bi);
    // std::cout << bi<<" \n";
    if (bi>=0){
      bins[bi].push_back(incoming_particles[i]);
    }
  }
  // std::cout << "End Rebin Particle: \n";
}

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

void move(particle_t& p,int rank, int numProcs, double size, vector<particle_t> &out_down_particles, vector<particle_t> &out_up_particles) {

  int cur_bin = -1;
  bin_ind(p, subdomain_start, subdomain_size, rank, numProcs, cur_bin);
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

  int new_bin = -1;
  bin_ind(p, subdomain_start, subdomain_size, rank, numProcs, new_bin);

  // if (new_bin == -2){
  //   std::cout << "-2"<< "\n";
  // }

  // rebinning is ok as long as you don't have to move procs
  // bin_id will just give -1, so have to check
  if (cur_bin != new_bin){
    vector<particle_t> bin = bins[cur_bin];
    for(int i = 0; i < bin.size(); ++i){
        if(&bin[i] == &p){
            bin.erase(bin.begin() + i);
            break;
        }
    }

    if ((new_bin < ((bin_dim_x - 1)*bin_dim)) && new_bin >= bin_dim){
      bins[new_bin].push_back(p);
    }

    // if you have to move one proc down
    else if (new_bin == -2){
      // std::cout << "-2" << "\n";
      out_down_particles.push_back(p);
    }

    // if you have to move one proc up
    else if (new_bin == -3) {
      // std::cout << "-3" << "\n";
      out_up_particles.push_back(p);
    }
  }
}

void send_out_down_particles(vector<particle_t> &out_down_particles,
                           int world_rank, int world_size) {
  // std::cout << "Begin send down: \n";
  // Send the data as an array of MPI_BYTEs to the next process.
  if (world_rank == world_size-1) return;
  MPI_Send(out_down_particles.data(),
           out_down_particles.size(), PARTICLE,
           world_rank + 1, 0, MPI_COMM_WORLD);
  // std::cout << out_down_particles->size() << "\n";
  // Clear the outgoing walkers list
  out_down_particles.clear();
  // std::cout << "End send down: \n";
}

void send_out_up_particles(vector<particle_t> &out_up_particles,
                           int world_rank, int world_size) {
  // std::cout << "Begin send up: \n";
  // Send the data as an array of MPI_BYTEs to the next process.
  if (world_rank == 0) return;
  MPI_Send(out_up_particles.data(),
           out_up_particles.size(), PARTICLE,
           world_rank - 1, 0, MPI_COMM_WORLD);
  // std::cout << out_up_particles->size() << "\n";
  // Clear the outgoing walkers list
  out_up_particles.clear();
  // std::cout << "End send up: \n";
}

void receive_above_particles(vector<particle_t> &incoming_particles,
                              int world_rank, int world_size) {
  // std::cout << "Begin receive above: \n";
  // Probe for new incoming particles
  MPI_Status status;
  // Receive from the process before you. If you are process zero, leave
  if (world_rank == 0) return;
  int incoming_rank = world_rank - 1;
  MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status); // change this to another send later to avoid deadlock
  // Resize your incoming particle  buffer based on how much data is being received

  int incoming_particles_size;
  MPI_Get_count(&status, PARTICLE, & incoming_particles_size);
  incoming_particles.resize(incoming_particles_size);
  MPI_Recv(incoming_particles.data(), incoming_particles_size,
           PARTICLE, incoming_rank, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  // std::cout << "End receive above: \n";
}

void receive_below_particles(vector<particle_t> &incoming_particles,
                              int world_rank, int world_size) {
  // std::cout << "Begin receive below: \n";
  // Probe for new incoming particles
  MPI_Status status;
  // Receive from the process after  you. If you are the last process, leave
  if (world_rank == world_size-1) return;
  int incoming_rank = world_rank + 1;
  MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);
  // Resize your incoming particle  buffer based on how much data is being received

  int incoming_particles_size;
  MPI_Get_count(&status, PARTICLE, & incoming_particles_size);
  incoming_particles.resize(incoming_particles_size);
  MPI_Recv(incoming_particles.data(), incoming_particles_size,
           PARTICLE, incoming_rank, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  // std::cout << "End receive below: \n";
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
  // std::cout << "Begin init_simulation: \n";
  bin_dim = floor(size / BINSIZE) + 1;

  // Distribute bins to processors
  decompose_domain(rank, num_procs);

  // Distribute particles to bins
  initialize_bins(parts, num_parts, subdomain_start, subdomain_size, rank, num_procs);
  // std::cout << "End init_simulation: \n";

  // incoming_particles.reserve(num_parts);
  // out_down_particles.reserve(num_parts);
  // out_up_particles.reserve(num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
  // std::cout << "Begin simulate_one_step: \n";
  // Run in serial
  for(int i = 0; i < num_parts; i++){
    parts[i].ax = parts[i].ay = 0;
  }

  for(int i = 1; i < bin_dim_x-1; ++i){ // only loop over non-ghost particles
        for(int j = 0; j < bin_dim; ++j){
            auto &bin = bins[i * bin_dim + j];
            const int bin_size = bin.size();

            for(int d = 0; d < 9; ++d){
                int neighbor_binx = i + directions[d][0];
                int neighbor_biny = j + directions[d][1];
                if(neighbor_binx < 0 or neighbor_binx >= bin_dim_x or neighbor_biny < 0 or neighbor_biny >= bin_dim)
                    continue;
                auto &neighbor_bin = bins[neighbor_binx * bin_dim + neighbor_biny];
                const int neighbor_size = neighbor_bin.size();
                for(int l = 0; l < bin_size; ++l){
                    particle_t cur = bin[l];
                    for(int k = 0; k < neighbor_size; ++k){
                        particle_t neighbor = neighbor_bin[k];
                        apply_force(cur, neighbor);
                    }
                }
            }
        }
    }
  // std::cout << "End simulate_one_step: \n";

  // std::cout << "Begin simulate_move: \n";
  // move all particles in bins (loop over bins maybe)
  for(int i = 1; i < bin_dim_x-1; ++i){ // only loop over non-ghost particles
    for(int j = 0; j < bin_dim; ++j){
      auto &bin = bins[i * bin_dim + j];
      const int bin_size = bin.size();
      for(int l = 0; l < bin_size; ++l){
        particle_t cur = bin[l];
        move(cur, rank, num_procs, size, out_down_particles, out_up_particles);
      }
    }
  }
  // std::cout << "End simulate_move: \n";

  // Send outgoing particles to new processors, receive from others
  // Idea here is move in one direction; every processor sends down, receives up
  // Then, every procesor sends up, received down

  // std::cout << "Begin MPI: \n";
  // Processors sending down, receiving up; even send first, odd receive first
  if (rank % 2 == 0){
    send_out_down_particles(out_down_particles, rank, num_procs);
    receive_above_particles(incoming_particles, rank, num_procs);
  }
  else{
    receive_above_particles(incoming_particles, rank, num_procs);
    send_out_down_particles(out_down_particles, rank, num_procs);
  }

  // std::cout << out_down_particles.size() << "\n";

  // Rebin received particles
  rebin_particles(incoming_particles, rank, num_procs);

  // Processors sending up, receiving down; odd send first, even receive first
  if (rank % 2 == 1){
    send_out_up_particles(out_up_particles, rank, num_procs);
    receive_below_particles(incoming_particles, rank, num_procs);
  }
  else{
    receive_below_particles(incoming_particles, rank, num_procs);
    send_out_up_particles(out_up_particles, rank, num_procs);
  }
  // std::cout << "End MPI: \n";

  // Rebin received particles
    rebin_particles(incoming_particles, rank, num_procs);
}

bool particle_sorter(particle_t& p1, particle_t& p2) {
  if (p1.id != p2.id) return p1.id < p2.id;
  else return p1.x < p2.x;

}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

  std::cout << "step1";
  vector<particle_t> local_parts; // get particles for each proc
  for(int i = 1; i < bin_dim_x-1; ++i){ // only loop over non-ghost particles
     for(int j = 0; j < bin_dim; ++j){
      auto &bin = bins[i * bin_dim + j];
      const int bin_size = bin.size();
      for(int l = 0; l < bin_size; ++l){
         particle_t cur = bin[l];
         local_parts.push_back(cur);
       }
     }
   }

  std::cout << "step2";
  int count = local_parts.size();
  
  // gather particles into proc 0
   particle_t* new_parts = NULL;
   int* counts = NULL;
   if (rank == 0) {
     new_parts = (particle_t *)malloc(sizeof(particle_t) * num_parts);
     counts = (int *)malloc(sizeof(int) * num_parts);
   }

   std::cout << "step3";
   MPI_Gather(&count, 1,MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

   std::cout << "step4";
   MPI_Gatherv(local_parts.data(), local_parts.size(),PARTICLE ,
                 parts, counts, 0, PARTICLE, 0, MPI_COMM_WORLD);

   // sort particles on proc 0
   if (rank ==0){
     sort(new_parts, new_parts+num_parts, &particle_sorter);
     parts = new_parts;
   }
   if (rank == 0) {
    free(new_parts);
    free(counts);
  }
}