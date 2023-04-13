#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include "butil.hpp"

struct HashMap {
    upcxx::atomic_domain<int> ad = upcxx::atomic_domain<int>({upcxx::atomic_op::compare_exchange,upcxx::atomic_op::load});

    std::vector <upcxx::global_ptr<kmer_pair>> global_data;
    std::vector <upcxx::global_ptr<int>> global_used;

    size_t global_size;
    size_t my_size;
    size_t offset;
    size_t my_start;
    int n_slot;
    int rank;
    
    size_t size() const noexcept;

    HashMap(size_t global_size);  
    ~HashMap();                           

    bool insert(const kmer_pair &kmer);
    bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

    // Helper functions
    int cax_global(uint64_t pos);
    bool slot_used(uint64_t pos);//, upcxx::global_ptr<int> tmp);
    void next_slot(uint64_t* global_slot);

    // Write and read to a logical data slot in the table.
  //  void write_slot(uint64_t rank, uint64_t pos, const kmer_pair& kmer); // changed
  //  kmer_pair read_slot(uint64_t rank, uint64_t pos); // changed

    // Request a slot or check if it's already used.
    //bool request_slot(uint64_t rank, uint64_t pos); // changed
    //bool slot_used(uint64_t rank, uint64_t pos); // changed
};

HashMap::HashMap(size_t global_size) {
    this->global_size = global_size;
    n_slot = upcxx::rank_n();
    rank = upcxx::rank_me();
    offset = (global_size + n_slot - 1) / n_slot;
    my_start = offset * rank;
    my_size = offset;
    if (rank == n_slot - 1) {
        my_size = global_size - (n_slot - 1) * offset;
    }
    global_data = std::vector<upcxx::global_ptr<kmer_pair>> (n_slot);
    global_used = std::vector<upcxx::global_ptr<int>> (n_slot);
    
    upcxx::global_ptr<kmer_pair> local_data = upcxx::new_array<kmer_pair>(my_size);
    upcxx::global_ptr<int> local_used = upcxx::new_array<int>(my_size);

    upcxx::future<upcxx::global_ptr<kmer_pair>> * data_fut = new upcxx::future<upcxx::global_ptr<kmer_pair>>[n_slot];
    upcxx::future<upcxx::global_ptr<int>> * used_fut = new upcxx::future<upcxx::global_ptr<int>>[n_slot];
    for (int i = 0; i < upcxx::rank_n(); i++) {
        data_fut[i] = upcxx::broadcast(local_data, i);
        used_fut[i] = upcxx::broadcast(local_used, i);
    }
    for (int i = 0; i < upcxx::rank_n(); i++) {
        global_data[i] = data_fut[i].wait();
        global_used[i] = used_fut[i].wait();
    }
}



bool HashMap::insert(const kmer_pair &kmer) {
  /*  uint64_t hash = kmer.hash();
    bool success = false;
    uint64_t probe = 0;
    do {
      uint64_t slot = hash % global_size;
      uint64_t rank = slot / n_slot;
      uint64_t pos = slot % n_slot;
        success = request_slot(rank, pos);
        if (success) {
            write_slot(rank, pos, kmer);
        }
    } while (!success && probe < global_size);
    return success;
}*/
uint64_t hash = kmer.hash();
uint64_t pos = hash % global_size;
uint64_t initial_pos = pos;
/*upcxx::global_ptr<int> ptr = global_data[pos/offset]+(pos%offset);
    if(ptr.is_local()){
      int *local_ptr = ptr.local();
      if(global_used[local_ptr])
    }else{*/
    //this is if you need to be global
    while (cax_global(pos) != 0) {
        next_slot(&pos);
        if (initial_pos == pos)
            return false;
    }
    upcxx::rput(kmer, global_data[pos/offset]+(pos%offset)).wait();
    return true;
}


bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t pos = hash % global_size;
    uint64_t initial_pos = pos;
    do {
        if (slot_used(pos)){
          break;
        }
        kmer_pair my_kmer = upcxx::rget(global_data[pos/offset]+(pos%offset)).wait();
        if (my_kmer.kmer == key_kmer) {
            val_kmer = my_kmer;
            return true;
        }
        next_slot(&pos);
    } while (pos != initial_pos);
    return false;
}

int HashMap::cax_global(uint64_t pos) {
    return ad.compare_exchange(global_used[pos/offset] + (pos%offset),
                                0, 1, std::memory_order_relaxed).wait();
}

bool HashMap::slot_used(uint64_t pos) {
    return ad.load(global_used[pos/offset] + (pos%offset), std::memory_order_relaxed).wait() == 0;
}

void HashMap::next_slot(uint64_t* global_slot) {
    *global_slot += 1;
    *global_slot %= global_size;
}

/*
bool HashMap::request_slot(uint64_t rank, uint64_t pos) { // changed
    if (global_used[rank+pos].is_local()) {
        int *used_local = global_used[rank+pos].local();
        if(used_local[pos] != 0) {
            return 0;
        } else {
            used_local[pos] = 1;
            return 1;
        } // race condition?
    } else {
        upcxx::future<int> res = ad.compare_exchange(global_used[rank] + pos, 0, 1, std::memory_order_relaxed); // sync issue?
        return res.wait();
    }
}
void HashMap::write_slot(uint64_t rank, uint64_t pos, const kmer_pair& kmer) { // changed
    if (global_data[rank+pos].is_local()) {
        kmer_pair *data_local = global_data[rank+pos].local();
        data_local[pos] = kmer;
    } else {
        //upcxx::rput(kmer, data_ptrs[rank] + pos);
        upcxx::rput(kmer, global_data[rank] + pos).wait(); // why?
    }
}
*/

HashMap::~HashMap() {
  this->ad.destroy();
}