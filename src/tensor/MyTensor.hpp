#pragma once

#include "BTensor.h"

class DramDataContainer;

class MyTensor : public BTensor {
public:
    MyTensor(std::string name, std::vector<uint32_t> dims, TensorType tensor_type, bool produced,
             AllocationScheme layout_scheme = MyAddressAllocator::allocation_scheme);

    ~MyTensor();

    std::string name;
    TensorType _tensor_type;
    AllocationScheme _layout_scheme;
    std::vector<uint32_t> _dims;

    uint32_t _precision;
    bool _is_transposed;

    // allocate_rows, physical address if use_virtual_memory is false, else virtual address with hash mapping
    uint32_t start_row;
    uint32_t end_row;

    uint32_t start_inner_row_loop;
    uint32_t start_middle_row_loop;
    uint32_t start_outer_row_loop;
    uint32_t start_col;

    uint32_t end_inner_row_loop;
    uint32_t end_middle_row_loop;
    uint32_t end_outer_row_loop;
    uint32_t end_col;

    std::vector<uint32_t> allocate_rows;
    std::vector<std::vector<uint32_t>> kv_cache_allocate_rows;
    uint32_t Cache_length;
    uint32_t Cache_capacity;

    std::vector<uint32_t> get_dims();
    void set_start_location();
    void set_end_location();
    addr_type get_addr(std::vector<uint32_t> indexes);
    std::vector<addr_type> get_all_addrs();
    void add_token();

    addr_type get_activation_addr(std::vector<uint32_t> indexes);
    addr_type get_weight_addr(std::vector<uint32_t> indexes);

    void set_transposed();
    void unset_transposed();

    std::vector<addr_type> get_addrs(std::vector<std::vector<uint32_t>> indexes);
    std::vector<addr_type> get_addrs_from_burst_offset(std::vector<uint32_t> burst_offsets);

    addr_type get_sequence_address(uint32_t burst_offset);
    addr_type get_DASH_2D_weight_address(uint32_t burst_offset);

    std::vector<addr_type> generate_addrs_based_on_indexes(std::vector<std::vector<uint32_t>> indexes);
    std::vector<addr_type> generate_addrs_based_on_indexes(std::vector<std::vector<uint32_t>> indexes,
                                                           AllocationScheme layout_scheme);
    std::vector<addr_type> generate_addrs_based_on_indexes_attention(std::vector<std::vector<uint32_t>> indexes, uint32_t head_index);

    AllocationScheme get_layout_scheme() const { return _layout_scheme; }

    // 2D Weight only: column-major layout, granularity_bytes (default 1KB) aligned chunk starts.
    // Sub-rectangle [k0,k1) x [n0,n1) with _dims = [K,N] (row k, col n). Returns one DRAM address per chunk
    // (burst containing chunk start), same mapping path as Activation via get_sequence_address.
    std::vector<addr_type> generate_weight_column_major_1kb_base_addrs(uint32_t k0, uint32_t n0, uint32_t k1, uint32_t n1, uint32_t granularity_bytes = 1024);

    std::vector<addr_type> generate_pim_comp_addrs(std::vector<uint32_t> start_rows, std::vector<uint32_t> indexes);
    std::vector<addr_type> generate_pim_comp_addrs_attention(uint32_t head_iteration_index, std::vector<std::vector<uint32_t>> indexes);

    void cache_append();

    void initial_data_container(DramDataContainer& data_container);
    void append_data_into_container(const std::vector<uint8_t>& data,
                                    DramDataContainer& data_container);
    uint64_t get_total_size();

};
