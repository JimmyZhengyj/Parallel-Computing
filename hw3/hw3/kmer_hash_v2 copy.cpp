#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>
#include <queue>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    // TODO: Dear Students,
    // Please remove this if statement, when you start writing your parallel implementation.
    /* if (upcxx::rank_n() > 1) { */
    /*     throw std::runtime_error("Error: parallel implementation not started yet!" */
    /*                              " (remove this when you start working.)"); */
    /* } */

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // Load factor of 0.5 and divide it by the number of processes
    size_t hash_table_size = n_kmers * (1.0 / 0.5) / (double)(upcxx::rank_n());
    HashMap hashmap(hash_table_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
                     n_kmers);
    }

    int rank_n = upcxx::rank_n();

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<kmer_pair> start_nodes;

    std::vector<std::vector<kmer_pair>> kvec(rank_n);

    for(kmer_pair &kmer: kmers){
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
        // Collect the kmer that belongs to the same process together
        kvec[hashmap.get_target_rank(kmer.kmer)].push_back(kmer);
    }

    int fs_n;
    std::vector<upcxx::future<>> fs;
    fs.reserve(rank_n);

    for(int t = 0; t < rank_n; ++t){
        if(kvec[t].empty())
            continue;
        upcxx::future<> fut = hashmap.insert_many(t, kvec[t]);
        fs.push_back(fut);
    }

    fs_n = fs.size();
    // Wait for each process to finish the insertion
    for(int i = 0; i < fs_n; ++i){
        fs[i].wait();
    }

    /* for (auto& kmer : kmers) { */
    /*     bool success = hashmap.insert(kmer); */
    /*     if (!success) { */
    /*         throw std::runtime_error("Error: HashMap is full!"); */
    /*     } */

    /*     if (kmer.backwardExt() == 'F') { */
    /*         start_nodes.push_back(kmer); */
    /*     } */
    /* } */
    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();

    auto start_read = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::list<kmer_pair>*>> cvec(rank_n);
    std::vector<std::vector<pkmer_t>> pvec(rank_n);
    std::list<std::list<kmer_pair>> contigs;


    /*
     * contig0-->
     * contig1-->
     * contig2-->
     * BFS expansion
     */
    std::queue<std::list<kmer_pair>*> q;
    // Put the start_kmers into the queue
    for(const auto& start_kmer: start_nodes){
        // Initialize the list
        contigs.push_back({start_kmer});
        // push the pointers inside the queue
        q.push(&contigs.back());
    }

    while(!q.empty()){
        // Clean up
        for(int t = 0; t < rank_n; ++t){
            cvec[t].clear();
            pvec[t].clear();
        }
        fs.clear();
        // BFS
        for(int i = q.size()-1; i >= 0; --i){
            auto contig_ptr = q.front();
            q.pop();

            // End
            if(contig_ptr->back().forwardExt() == 'F')
                continue;

            // Need to keep expanding this contig later
            q.push(contig_ptr);

            // Put the next_kmer to the corresponding target process
            pkmer_t key_kmer = contig_ptr->back().next_kmer();
            int target = hashmap.get_target_rank(key_kmer);

            // `key_kmer`'s information should be inserted into `contig_ptr`'s back
            cvec[target].push_back(contig_ptr);
            pvec[target].push_back(key_kmer);
        }
        // Initiate RPC for each target rank
        for(int t = 0; t < rank_n; ++t){
            // if there is nothing for this rank, continue
            if(pvec[t].empty()) continue;

            // batch find
            upcxx::future<std::vector<FB>> fut = hashmap.find_many(t, pvec[t]);

            auto &contig_ptrs = cvec[t];

            // **Register a callback**
            // so that after receiving the vector<FB>
            // it will immediately append these kmer into the back of each corresponding contig
            upcxx::future<> fut2 = fut.then([&contig_ptrs](const std::vector<FB> &ret_v){
                int n = ret_v.size();
                for(int i = 0; i < n; ++i){
                    std::list<kmer_pair> *contig_ptr = contig_ptrs[i];
                    auto &back = contig_ptr->back();
                    const FB &fb = ret_v[i];

                    // append this kmer
                    contig_ptr->push_back({});
                    auto &new_back = contig_ptr->back();
                    new_back.kmer = back.next_kmer();
                    new_back.fb_ext[0] = fb.b;
                    new_back.fb_ext[1] = fb.f;
                }
            });

            // append this future so that we can wait it later
            fs.push_back(fut2);
        }
        // Wait until all finished
        fs_n = fs.size();
        for(int i = 0; i < fs_n; ++i){
            fs[i].wait();
        }
    }

    /* for (const auto& start_kmer : start_nodes) { */
    /*     contigs.push_back({}); */
    /*     auto &contig = contigs.back(); */
    /*     contig.push_back(start_kmer); */
    /*     while (contig.back().forwardExt() != 'F') { */
    /*         kmer_pair kmer; */
    /*         bool success = hashmap.find(contig.back().next_kmer(), kmer); */
    /*         if (!success) { */
    /*             throw std::runtime_error("Error: k-mer not found in hashmap."); */
    /*         } */
    /*         contig.push_back(kmer); */
    /*     } */
    /* } */

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(), read.count(),
               insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}
