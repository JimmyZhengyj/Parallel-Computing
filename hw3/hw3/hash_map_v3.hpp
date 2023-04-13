#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include "butil.hpp"

struct HashMap {

    // std::vector<kmer_pair> data;
    // std::vector<int> used;

    std::vector <upcxx::global_ptr<kmer_pair>> data;
    std::vector <upcxx::global_ptr<int>> used;

    upcxx::atomic_domain<int> ad = upcxx::atomic_domain<int>({upcxx::atomic_op::compare_exchange,upcxx::atomic_op::load});
    
    size_t offset;
    int n_slot;

    size_t global_size;

    size_t size() const noexcept;

    HashMap(size_t global_size);                            

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Check if the slot is used
    bool slot_used(uint64_t slot);
    // Read logical data slot in the table
    upcxx::global_ptr<kmer_pair> read_slot_data(uint64_t slot);
    upcxx::global_ptr<int> read_slot_used(uint64_t slot);
    // Move to the next slot
    void next_slot(uint64_t* global_slot);

    // // Write and read to a logical data slot in the table.
    // void write_slot(uint64_t slot, const kmer_pair& kmer);
    // kmer_pair read_slot(uint64_t slot);

    // // Request a slot or check if it's already used.
    // bool request_slot(uint64_t slot);
    // bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t global_size) {

    // my_size = size;
    // data.resize(size);
    // used.resize(size, 0);

    this->global_size = global_size;

    n_slot = upcxx::rank_n();
    offset = (global_size - 1) / n_slot + 1;

    data = std::vector<upcxx::global_ptr<kmer_pair>>(n_slot);
    used = std::vector<upcxx::global_ptr<int>>(n_slot);

    size_t local_size = offset;
    if (n_slot - 1 == upcxx::rank_me()) {
        local_size = global_size - (n_slot - 1) * offset;
    }
    
    upcxx::global_ptr<kmer_pair> local_data = upcxx::new_array<kmer_pair>(local_size);
    upcxx::global_ptr<int> local_used = upcxx::new_array<int>(local_size);

    upcxx::future<upcxx::global_ptr<kmer_pair>> * data_fut = new upcxx::future<upcxx::global_ptr<kmer_pair>>[n_slot];
    upcxx::future<upcxx::global_ptr<int>> * used_fut = new upcxx::future<upcxx::global_ptr<int>>[n_slot];

    for (int i = 0; i < n_slot; i++) {
        data_fut[i] = upcxx::broadcast(local_data, i);
        used_fut[i] = upcxx::broadcast(local_used, i);
    }

    for (int i = 0; i < n_slot; i++) {
        data[i] = data_fut[i].wait();
        used[i] = used_fut[i].wait();
    }
}

bool HashMap::insert(const kmer_pair &kmer) {
    uint64_t hash = kmer.hash();
    // uint64_t probe = 0;
    // bool success = false;
    // do {
    //     uint64_t slot = (hash + probe++) % size();
    //     success = request_slot(slot);
    //     if (success) {
    //         write_slot(slot, kmer);
    //     }
    // } while (!success && probe < size());
    // return success;

    uint64_t slot = hash % global_size;
    uint64_t initial_pos = slot;

    int exchanged = ad.compare_exchange(read_slot_used(slot),0, 1, std::memory_order_relaxed).wait();

    while (exchanged != 0) {
        next_slot(&slot);
        if (initial_pos == slot)
            return false;
        exchanged = ad.compare_exchange(read_slot_used(slot),0, 1, std::memory_order_relaxed).wait();
    }
    upcxx::rput(kmer, read_slot_data(slot)).wait();
    return true;
}

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
    uint64_t hash = key_kmer.hash();
    // uint64_t probe = 0;
    // bool success = false;
    // do {
    //     uint64_t slot = (hash + probe++) % size();
    //     if (slot_used(slot)) {
    //         val_kmer = read_slot(slot);
    //         if (val_kmer.kmer == key_kmer) {
    //             success = true;
    //         }
    //     }
    // } while (!success && probe < size());
    // return success;

    uint64_t slot = hash % global_size;
    uint64_t initial_pos = slot;

    do {
        if (slot_used(slot)){
          break;
        }
        kmer_pair my_kmer = upcxx::rget(read_slot_data(slot)).wait();
        if (my_kmer.kmer == key_kmer) {
            val_kmer = my_kmer;
            return true;
        }
        next_slot(&slot);
    } while (slot != initial_pos);
    return false;
}

bool HashMap::slot_used(uint64_t slot) {
    return ad.load(read_slot_used(slot), std::memory_order_relaxed).wait() == 0;
}

upcxx::global_ptr<kmer_pair> HashMap::read_slot_data(uint64_t slot) { 
    return data[slot/offset]+(slot%offset); 
}

upcxx::global_ptr<int> HashMap::read_slot_used(uint64_t slot) { 
    return used[slot/offset]+(slot%offset); 
}

void HashMap::next_slot(uint64_t* global_slot) {
    *global_slot = (*global_slot + 1) % global_size;
}
