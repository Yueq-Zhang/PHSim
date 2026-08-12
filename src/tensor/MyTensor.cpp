#include "MyTensor.hpp"

#include "../DRAM/Dram.h"
#include "../DRAM/DramDataContainer.h"
#include "fmt/format.h"
#include <numeric>

namespace {
class ScopedAllocationScheme {
public:
    explicit ScopedAllocationScheme(AllocationScheme scheme)
        : saved_scheme(MyAddressAllocator::allocation_scheme) {
        MyAddressAllocator::allocation_scheme = scheme;
    }

    ~ScopedAllocationScheme() {
        MyAddressAllocator::allocation_scheme = saved_scheme;
    }

private:
    AllocationScheme saved_scheme;
};
}

MyTensor::MyTensor(std::string name, std::vector<uint32_t> dims, TensorType tensor_type, bool produced, AllocationScheme layout_scheme) {
    _name = name;
    _dims = dims;
    _tensor_type = tensor_type;
    _layout_scheme = layout_scheme;
    _precision = MyAddressAllocator::get_precision(_tensor_type);
    _produced = produced;

    Cache_length = 0;
    Cache_capacity = 0;

    spdlog::info("Initial Tensor: {}", name);
    ScopedAllocationScheme scoped_scheme(_layout_scheme);

    if (tensor_type == TensorType::WGT) {  // Weight Tensor allocation 1D and 2D
        // Separate arrange 1-D and 2-D weights
        _precision = MyAddressAllocator::precision_weight;
        set_start_location();
        if (_dims.size()==1) {
            allocate_rows = MyAddressAllocator::weight_allocate_1D(_dims, _precision);
        }
        else if (_dims.size()==2) {
            allocate_rows = MyAddressAllocator::weight_allocate_2D(_dims, _precision);
        }
        else {
            throw std::runtime_error("The Weight Tensor only Support 1-D or 2-D Type");
        }
        set_end_location();
    }
    else if (tensor_type == TensorType::ACT) {
        _precision = MyAddressAllocator::precision_activation;
        set_start_location();
        allocate_rows = MyAddressAllocator::activation_allocate(_dims, _precision);
        set_end_location();
    }
    else if (tensor_type == TensorType::KCache) {
        _precision = MyAddressAllocator::precision_cache;
        kv_cache_allocate_rows = MyAddressAllocator::kcache_allocate(_dims, _precision);
        Cache_length = _dims[0];        // current token lenght = dims[0]
        // Check if kv_cache_allocate_rows is empty before accessing
        uint32_t allocate_rows_size = kv_cache_allocate_rows.empty() ? 0 : kv_cache_allocate_rows[0].size();
        if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
            Cache_capacity = allocate_rows_size * MyAddressAllocator::KCache_rows_per_allocation;
        }
        else if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU) {
            Cache_capacity = std::floor(static_cast<double>(allocate_rows_size * MyAddressAllocator::total_banks) / MyAddressAllocator::h_kv) * MyAddressAllocator::kv_cache_entry_size;
        }


        // assert(_dims[1] == MyAddressAllocator::h_kv * MyAddressAllocator::d_k);
    }
    else if (tensor_type == TensorType::VCache) {
        _precision = MyAddressAllocator::precision_cache;
        kv_cache_allocate_rows = MyAddressAllocator::vcache_allocate(_dims, _precision);
        Cache_length = _dims[0];
        uint32_t rows_size = kv_cache_allocate_rows.empty() ? 0 : kv_cache_allocate_rows[0].size();
        Cache_capacity = rows_size * MyAddressAllocator::VCache_rows_per_allocation;
        // assert(_dims[1] == MyAddressAllocator::h_kv * MyAddressAllocator::d_k);
    }
    else {
        throw std::invalid_argument("tensor type not supported");
    }

    // if ()
    // initial_data_continer();
}

void MyTensor::cache_append() {
    spdlog::info("Before allocated, current Cache Capacity is {}", Cache_capacity);
    Cache_capacity += MyAddressAllocator::kvcache_append(&kv_cache_allocate_rows, _tensor_type);
    spdlog::info("After Allocated, the Cache Capacity is {}", Cache_capacity);
}


std::vector<uint32_t> MyTensor::get_dims() {
    return _dims;
}

uint64_t MyTensor::get_total_size() {
    uint64_t total_size = _precision;
    for (auto dim : _dims) {
        total_size = total_size * dim;
    }
    return total_size;
}


void MyTensor::set_start_location() {
    if (_tensor_type == TensorType::ACT){
        start_row = MyAddressAllocator::act_row;
        start_inner_row_loop = MyAddressAllocator::act_inner_row_loop;
        start_middle_row_loop = MyAddressAllocator::act_middle_row_loop;
        start_outer_row_loop = MyAddressAllocator::act_outer_row_loop;
        start_col = MyAddressAllocator::act_column;
    }
    else if (_tensor_type==TensorType::WGT){
        if (_dims.size()==1) {
            start_row = MyAddressAllocator::base_row_1D_weight;
            start_inner_row_loop = MyAddressAllocator::base_inner_row_loop_1D_weight;
            start_middle_row_loop = MyAddressAllocator::base_middle_row_loop_1D_weight;
            start_outer_row_loop = MyAddressAllocator::base_outer_row_loop_1D_weight;
            start_col = MyAddressAllocator::base_column_1D_weight;
        }
        else {
            start_row = MyAddressAllocator::base_row;
            start_inner_row_loop = MyAddressAllocator::base_inner_row_loop;
            start_middle_row_loop = MyAddressAllocator::base_middle_row_loop;
            start_outer_row_loop = MyAddressAllocator::base_outer_row_loop;
            start_col = MyAddressAllocator::base_column;
        }
    }
}

void MyTensor::set_end_location() {
    if (_tensor_type == TensorType::ACT) {
        end_row = MyAddressAllocator::act_row;
        end_inner_row_loop = MyAddressAllocator::act_inner_row_loop;
        end_middle_row_loop = MyAddressAllocator::act_middle_row_loop;
        end_outer_row_loop = MyAddressAllocator::act_outer_row_loop;
        end_col = MyAddressAllocator::act_column;
    }
    else if (_tensor_type==TensorType::WGT) {
        if (_dims.size()==1) {
            end_row = MyAddressAllocator::base_row_1D_weight;
            end_inner_row_loop = MyAddressAllocator::base_inner_row_loop_1D_weight;
            end_middle_row_loop = MyAddressAllocator::base_middle_row_loop_1D_weight;
            end_outer_row_loop = MyAddressAllocator::base_outer_row_loop_1D_weight;
            end_col = MyAddressAllocator::base_column_1D_weight;
        }
        else {
            end_row = MyAddressAllocator::base_row;
            end_inner_row_loop = MyAddressAllocator::base_inner_row_loop;
            end_middle_row_loop = MyAddressAllocator::base_middle_row_loop;
            end_outer_row_loop = MyAddressAllocator::base_outer_row_loop;
            end_col = MyAddressAllocator::base_column;
        }
    }
    else {
        end_row = MyAddressAllocator::base_row;
        end_inner_row_loop = MyAddressAllocator::base_inner_row_loop;
        end_middle_row_loop = MyAddressAllocator::base_middle_row_loop;
        end_outer_row_loop = MyAddressAllocator::base_outer_row_loop;
        end_col = MyAddressAllocator::base_column;
    }
}


addr_type MyTensor::get_addr(std::vector<uint32_t> indexes) {
    ast(_dims.size() == indexes.size());

    addr_type addr = 0;
    if (_tensor_type == TensorType::ACT or _tensor_type == TensorType::PSUM) {
        addr = get_activation_addr(indexes);
    }
    else if (_tensor_type == TensorType::WGT) {
        addr = get_weight_addr(indexes);
    }
    else if (_tensor_type == TensorType::KCache) {

    }
    else if (_tensor_type == TensorType::VCache) {

    }
    else {
        throw std::runtime_error("tensor type not supported");
    }

    return addr;
}

addr_type MyTensor::get_weight_addr(std::vector<uint32_t> indexes) {
    if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
        // weight
    }
    else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH){

    }
    else if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {

    }

    return 0;
}


addr_type MyTensor::get_activation_addr(std::vector<uint32_t> indexes) {
    uint64_t size;
    if (_dims.size() == 1) {
        size = indexes[0] * _precision;  // 1D
    }
    else {
        size = (indexes[0] * _dims[1] + indexes[1]) * _precision; // 2D activation
    }

    auto burst_offset = size / MyAddressAllocator::memory_burst_size;
    auto data_offset =  size % MyAddressAllocator::memory_burst_size;

    auto address = MyAddressAllocator::get_sequence_address(
        burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);

    return address;
}


std::vector<addr_type> MyTensor::get_all_addrs() {
    // spdlog::info("get_all_addrs of tensor with size {}", _dims);
    std::vector<addr_type> addrs;

    if (_tensor_type == TensorType::ACT or _tensor_type == TensorType::PSUM) {

        uint64_t size = _precision;
        for (auto dim : _dims) {
            size *= dim;
        }
        uint32_t burst_times = size / MyAddressAllocator::memory_burst_size;
        std::vector<uint32_t> burst_offsets;
        burst_offsets.reserve(burst_times);
        for (int i = 0; i < burst_times; i++) {
            burst_offsets.push_back(i);
        }
        addrs = get_addrs_from_burst_offset(burst_offsets);
    }
    else if (_tensor_type == TensorType::WGT) {
        if (_dims.size() == 1 or MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
            uint64_t size = _precision;
            for (auto dim : _dims) {
                size *= dim;
            }
            uint32_t burst_times = size / MyAddressAllocator::memory_burst_size;
            std::vector<uint32_t> burst_offsets;
            burst_offsets.reserve(burst_times);
            for (int i = 0; i < burst_times; i++) {
                burst_offsets.push_back(i);
            }
            addrs = get_addrs_from_burst_offset(burst_offsets);
        }
        else {
            throw std::runtime_error("Not supported");
        }
    }
    else if (_tensor_type == TensorType::KCache) {

        throw std::runtime_error("Not supported");
    }
    else if (_tensor_type == TensorType::VCache) {

        throw std::runtime_error("Not supported");
    }
    else {
        throw std::runtime_error("tensor type not supported");
    }

    return addrs;
}


void MyTensor::add_token() {
    spdlog::info("add_token");
}

void MyTensor::set_transposed() {
    assert(_tensor_type == TensorType::ACT || _tensor_type == TensorType::WGT);
    _is_transposed = true;
}

void MyTensor::unset_transposed() {
    assert(_tensor_type == TensorType::ACT || _tensor_type == TensorType::WGT);
    _is_transposed = false;
}


std::vector<addr_type> MyTensor::get_addrs(std::vector<std::vector<uint32_t>> indexes) {

    std::vector<addr_type> addrs;
    // spdlog::info("The dimension of current tensor is {}", _dims.size());
    // spdlog::info("the number of operation indexes is {} ", indexes.size());
    std::vector<uint64_t> offsets;

    if (_tensor_type == TensorType::ACT or _tensor_type == TensorType::PSUM) {
        if (_dims.size() == 1) {
            for (int i = 0; i < indexes.size(); ++i) {
                offsets.push_back(indexes[i][0]);
            }
        }
        else {
            for (int i = 0; i < indexes.size(); ++i) {
                offsets.push_back((indexes[i][0] * _dims[1] + indexes[i][1]));
            }
        }
    }
    else if (_tensor_type == TensorType::WGT) {
        if (_dims.size() == 1) {
            for (int i = 0; i < indexes.size(); ++i) {
                offsets.push_back(indexes[i][0]);
            }
        }
        else {
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
                for (int i = 0; i < indexes.size(); ++i) {
                    offsets.push_back((indexes[i][0] * _dims[1] + indexes[i][1]));
                }
            }
            else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                throw std::runtime_error("The DASH scheme not support generate addr based on offset");
                return addrs;
            }
            else {
                throw std::runtime_error("Other allocation scheme not supported");
            }
        }
    }
    else if (_tensor_type == TensorType::KCache) {

    }
    else if (_tensor_type == TensorType::VCache) {

    }
    else {
        throw std::runtime_error("Tensor type not supported");
    }


    uint32_t data_unit = MyAddressAllocator::memory_burst_size/_precision;
    uint32_t burst_offset = 0;

    for (int i = 0; i < indexes.size(); ++i) {
        if (i == 0 or burst_offset != offsets[i] / data_unit) {
            burst_offset = offsets[i] / data_unit;
            addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
            // spdlog::info("Generate addr for activation is {}, with burst_offset {}", addr, burst_offset);
            addrs.push_back(addr);
        }
    }
    return addrs;
}



std::vector<addr_type> MyTensor::get_addrs_from_burst_offset(std::vector<uint32_t> burst_offsets) {
    std::vector<addr_type> addrs;

    if (_tensor_type == TensorType::ACT or _tensor_type == TensorType::PSUM) {
        for (uint32_t i = 0; i < burst_offsets.size(); ++i) {
            addr_type addr = MyAddressAllocator::get_sequence_address(burst_offsets[i], start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, allocate_rows, start_col);
            addrs.push_back(addr);
        }
    }
    else if (_tensor_type == TensorType::WGT) {
        if (_dims.size() == 1 or MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
            for (uint32_t i = 0; i < burst_offsets.size(); ++i) {
                addr_type addr = MyAddressAllocator::get_sequence_address(burst_offsets[i], start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
                addrs.push_back(addr);
            }
        }
        else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {

        }

    }
    else if (_tensor_type == TensorType::KCache) {

    }

    return addrs;
}



addr_type MyTensor::get_sequence_address(uint32_t burst_offset) {

    start_row = MyAddressAllocator::act_row;
    start_inner_row_loop = MyAddressAllocator::act_inner_row_loop;
    start_middle_row_loop = MyAddressAllocator::act_middle_row_loop;
    start_outer_row_loop = MyAddressAllocator::act_outer_row_loop;
    start_col = MyAddressAllocator::act_column;

    // row number and column offset
    uint32_t col_offset = (start_col + burst_offset) % MyAddressAllocator::BL_num_per_row;
    uint32_t allocate_row_num = (start_col + burst_offset) / MyAddressAllocator::BL_num_per_row;

    // inner_row_loop offset
    uint32_t inner_row_offset = (start_inner_row_loop + allocate_row_num) % MyAddressAllocator::row_loop_size[MyAddressAllocator::inner_row_loop];
    uint32_t allocate_row_middle = (start_inner_row_loop + allocate_row_num) / MyAddressAllocator::row_loop_size[MyAddressAllocator::inner_row_loop];

    // middle_row_loop offset
    uint32_t middle_row_offset = (start_middle_row_loop + allocate_row_middle) % MyAddressAllocator::row_loop_size[MyAddressAllocator::middle_row_loop];
    uint32_t allocate_row_outer = (start_middle_row_loop + allocate_row_middle) / MyAddressAllocator::row_loop_size[MyAddressAllocator::middle_row_loop];

    // outer_row_loop offset
    uint32_t outer_row_offset = (start_outer_row_loop + allocate_row_outer) % MyAddressAllocator::row_loop_size[MyAddressAllocator::outer_row_loop];
    uint32_t allocate_row = (start_outer_row_loop + allocate_row_outer) / MyAddressAllocator::row_loop_size[MyAddressAllocator::outer_row_loop];

    // row offset
    uint32_t row_index = allocate_rows[allocate_row];

    auto addr =  MyAddressAllocator::make_address(outer_row_offset, middle_row_offset, inner_row_offset, row_index, col_offset, 0);
    return addr;
}


addr_type MyTensor::get_DASH_2D_weight_address(uint32_t burst_offset) {
    uint32_t interleave_banks = MyAddressAllocator::interleaved_banks_per_tile;
    uint32_t tiles_per_iteration =  MyAddressAllocator::allocated_tiles_per_iteration;

    // Directly calculate the corresponding position based on burst_offset; incorporate offset as an ordered mapping,
    // first determining which tile it belongs to by calculating from the offset, then computing the tile's position in the memory system
    uint32_t burst_num_per_tile = (allocate_rows[1] - allocate_rows[0]) * MyAddressAllocator::interleaved_banks_per_tile * MyAddressAllocator::BL_num_per_row;
    uint32_t burst_index_in_tile = burst_offset % burst_num_per_tile;  // The position of the current burst offset within the current tile
    uint32_t tile_index = burst_offset / burst_num_per_tile;  // Calculate the position of the tile based on the default mapping allocation order to obtain the Rank BankGroup Bank
    uint32_t tile_iteration_index = tile_index / MyAddressAllocator::allocated_tiles_per_iteration;  // get the starting row is obtained based on the iteration number
    uint32_t tile_iteration_offset = tile_index % MyAddressAllocator::allocated_tiles_per_iteration; // Obtain bank index based on the offset of iterated allocation

    // row
    uint32_t burst_row_index = allocate_rows[tile_iteration_index] + burst_index_in_tile / (MyAddressAllocator::interleaved_banks_per_tile * MyAddressAllocator::BL_num_per_row);
    uint32_t burst_offset_in_row = burst_index_in_tile % (MyAddressAllocator::interleaved_banks_per_tile * MyAddressAllocator::BL_num_per_row);

    // rank-bankgroup-bank
    uint32_t rank_index = tile_iteration_offset / (MyAddressAllocator::bankgroups * MyAddressAllocator::banks / MyAddressAllocator::interleaved_banks_per_tile);
    uint32_t rank_offset = tile_iteration_offset % (MyAddressAllocator::bankgroups * MyAddressAllocator::banks / MyAddressAllocator::interleaved_banks_per_tile);

    uint32_t bank_index = rank_offset / (MyAddressAllocator::bankgroups / MyAddressAllocator::interleaved_banks_per_tile);
    uint32_t bankgroup_index = rank_offset % (MyAddressAllocator::bankgroups / MyAddressAllocator::interleaved_banks_per_tile);

    // std::vector<std::vector<uint32_t>> tile_bank_location_lookuptable;
    spdlog::info("Current burst offset {} belongs to {}th tile, corresponding to the {} iteration {} offset tile allocation", burst_offset, tile_index, tile_iteration_index, tile_iteration_offset);
    spdlog::info("The begin of the tile is {} Row, {} Rank, {} Bank, {} Bankgroup", burst_row_index, rank_index, bank_index, bankgroup_index);  //

    // outer_row_offset = rank
    // middle_row_offset = bank
    // inner_row_offset = bankgroup
    //  auto addr =  MyAddressAllocator::make_address(outer_row_offset, middle_row_offset, inner_row_offset, burst_row_index, col_offset, 0);
    return 0;
}


std::vector<addr_type> MyTensor::generate_addrs_based_on_indexes(std::vector<std::vector<uint32_t>> indexes) {
    if (indexes.size() == 0) {
        return {};
    }
    assert(indexes[0].size() == _dims.size());

    std::vector<addr_type> addrs;
    std::vector<uint32_t> burst_offsets;
    if (_dims.size() == 1) {
        // if 1D Tensor, Row major, data are stored in mutiple columns, all the Scheme are Same.
        for (auto index : indexes) {
            uint32_t burst_offset = index[0] * _precision / MyAddressAllocator::memory_burst_size;
            // get burst offset
            if (burst_offsets.empty()) {
                burst_offsets.push_back(burst_offset);
            }
            else {
                if (burst_offsets.back() != burst_offset) {
                    burst_offsets.push_back(burst_offset);
                }
            }
        }

        if (_tensor_type==TensorType::ACT) {  // ACT Row major
            for (auto burst_offset : burst_offsets) {
                addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
                addrs.push_back(addr);
            }
        }
        else if (_tensor_type==TensorType::WGT){
            for (auto burst_offset : burst_offsets) {
                addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
                addrs.push_back(addr);
            }
        }
        else {
            throw std::runtime_error("Tensor type not supported");
        }
    }
    else {  // _dims.size() == 2  // For Activation
        if (_tensor_type == TensorType::ACT or _tensor_type == TensorType::PSUM) {
            if (false) { // MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS
                if (_dims.size() == 2) {
                    if (_dims[0] == 1) {
                        // 2D Activation Tensor Vector, Row Major
                        for (auto index : indexes) {
                            uint32_t burst_offset = (index[0] * _dims.back() + index.back()) * _precision / MyAddressAllocator::dram_burst_size;
                            if (burst_offsets.empty()) {
                                burst_offsets.push_back(burst_offset);
                            }
                            else {
                                if (burst_offsets.back() != burst_offset) {
                                    burst_offsets.push_back(burst_offset);
                                }
                            }
                        }
                    }
                    else {  // 2D Activation Tensor Matrix, Row Major arrangement split
                        uint32_t begin_row = indexes.front()[0];
                        uint32_t begin_col = indexes.front()[1];
                        uint32_t end_row = indexes.back()[0];
                        uint32_t end_col = indexes.back()[1];

                        uint32_t burst_offsets_per_row = _dims[1] * _precision / MyAddressAllocator::dram_burst_size;  // 当前矩阵的每??行burst大小
                        uint32_t col_burst_begin = begin_col * _precision / MyAddressAllocator::dram_burst_size;
                        uint32_t col_burst_end = end_col * _precision / MyAddressAllocator::dram_burst_size;

                        uint32_t current_burst_offset = 0;
                        for (uint32_t row_offset = begin_row; row_offset <= end_row; row_offset++) {
                            for (uint32_t col_burst_offset = col_burst_begin; col_burst_offset <= col_burst_end; col_burst_offset++) {
                                current_burst_offset = row_offset * burst_offsets_per_row + col_burst_offset;
                                burst_offsets.push_back(current_burst_offset);
                            }
                        }
                    }
                }
                else if (_dims.size() == 3) {  // 3 dims operation
                    // split the activation tensor into multiple channels, each channel is stored in mutiple columns
                }
                for (auto burst_offset : burst_offsets) {
                    addrs.push_back(MyAddressAllocator::get_sequence_address_ianus(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col));
                }
                // MyAddressAllocator::check_addrs(addrs);
            }
            else {  // NPU or DASH Activation Allocation
                if (_dims.size() == 2) {
                    if (_dims[0] == 1) {  // 2D Activation Tensor Vector, Row Major
                        for (auto index : indexes) {
                            uint32_t burst_offset = (index[0] * _dims.back() + index.back()) * _precision / MyAddressAllocator::memory_burst_size;
                            if (burst_offsets.empty()) {
                                burst_offsets.push_back(burst_offset);
                            }
                            else {
                                if (burst_offsets.back() != burst_offset) {
                                    burst_offsets.push_back(burst_offset);
                                }
                            }
                        }
                    }
                    else { // 2D Activation Tensor Matrix, Channel Parallel Row Major, with a split
                        uint32_t column_slice_size = MyAddressAllocator::act_column_slice_size;
                        uint32_t dram_burst_num = std::ceil(static_cast<double>(MyAddressAllocator::dram_burst_size)/_precision);

                        uint32_t burst_times_per_col_slice = column_slice_size / dram_burst_num;
                        uint32_t last_column_slice_length = _dims[1] % column_slice_size;
                        uint32_t last_column_slice_burst_times = (last_column_slice_length == 0) ? burst_times_per_col_slice : std::ceil(static_cast<double>(last_column_slice_length) / dram_burst_num);

                        uint32_t start_row = indexes.front()[0];
                        uint32_t start_col = indexes.front()[1];
                        uint32_t end_row = indexes.back()[0];
                        uint32_t end_col = indexes.back()[1];

                        uint32_t start_col_slice = std::floor(start_col/column_slice_size);
                        uint32_t end_col_slice = std::ceil(end_col/column_slice_size);

                        std::unordered_map<uint32_t, std::vector<uint32_t>> column_slice_burst_offsets;
                        for (uint32_t i = start_col_slice; i <= end_col_slice; ++i) {
                            column_slice_burst_offsets[i] = std::vector<uint32_t>();
                        }

                        assert(column_slice_size % dram_burst_num == 0);
                        for (uint32_t col_burst_index = start_col/dram_burst_num; col_burst_index <= end_col/dram_burst_num; col_burst_index++) {
                            uint32_t col_slice_id = col_burst_index / burst_times_per_col_slice;
                            column_slice_burst_offsets[col_slice_id].push_back(col_burst_index);
                        }

                        uint32_t column_slice_burst_offset;     // burst offset for previous column slice
                        uint32_t current_slice_row_burst_times; // burst offset of current column slice
                        // 下面是加载过程，按照行方向进行加载，如果遇到了column slice边界在后面换??
                        for (uint32_t col_slice_id = start_col_slice; col_slice_id <= end_col_slice; ++col_slice_id) {
                            column_slice_burst_offset = col_slice_id * std::ceil(static_cast<double>(_dims[0]) / MyAddressAllocator::dram_channels) * burst_times_per_col_slice; // 前序切片??跳过offset数量
                            if (col_slice_id == end_col_slice) {
                                current_slice_row_burst_times = last_column_slice_burst_times; // 当前slice对应的是??后一个列切片，其长度小于等于column_slice
                            }
                            else {
                                current_slice_row_burst_times = burst_times_per_col_slice; // 当前slice不是??后一个列切片，其长度等于column_slice
                            }
                            for (uint32_t row_iteration_index = start_row/MyAddressAllocator::dram_channels; row_iteration_index <= end_row/MyAddressAllocator::dram_channels; ++row_iteration_index) { // 当前切片对应的行??
                                uint32_t row_iteration_offset = row_iteration_index * current_slice_row_burst_times;  // row iteration × row offset
                                for (auto col_offset:column_slice_burst_offsets[col_slice_id]) {
                                    burst_offsets.push_back(column_slice_burst_offset + row_iteration_offset + col_offset);
                                }
                            }
                        }
                        assert(burst_offsets.size() == std::ceil(static_cast<double>(end_col-start_col)/dram_burst_num) * std::ceil(static_cast<double>(end_row-start_row)/MyAddressAllocator::dram_channels));
                    }
                }
                else if(_dims.size() == 3) {
                    // todo::这里的坐标是head, token, embd, 重新确定计算方式 , 这部分内容后续也??要进??步实??
                    if (_dims[1] == 1) { // single token-multi-head
                        for (auto index : indexes) {
                            uint32_t burst_offset = ((index[0] * _dims[1] + index[1]) * _dims.back() + index.back()) * _precision / MyAddressAllocator::memory_burst_size;
                            if (burst_offsets.empty()) {
                                burst_offsets.push_back(burst_offset);
                            }
                            else {
                                if (burst_offsets.back() != burst_offset) {
                                    burst_offsets.push_back(burst_offset);
                                }
                            }
                        }
                    }
                    else {  // multi-token-multi-head
                        // Device each row
                        uint32_t column_slice_size = MyAddressAllocator::act_column_slice_size; // Config::system_config.model_n_embd;
                        uint32_t column_slice_index = 0;
                        uint32_t column_slice_offset = 0;
                        // spdlog::info("Current DRAM Channel Number is {}", MyAddressAllocator::dram_channels);
                        uint32_t row_iteration_index = 0;
                        uint32_t row_iteration_border = 0;
                        uint32_t row_index = 0;

                        for (auto index : indexes) {  // each 3 dimension index, the outer is head index, must be separable
                            // To realize the column slice of row major activation tensor
                            column_slice_index = std::floor(index[2]/column_slice_size);
                            column_slice_offset = index[2] % column_slice_size;

                            if (burst_offsets.empty()) {
                                // The first burst must be available, and initial the
                                row_iteration_index = std::floor( (index[0] * _dims[1] + index[1]) / MyAddressAllocator::dram_channels);
                                row_iteration_border = (row_iteration_index + 1) * MyAddressAllocator::dram_channels;
                                row_index = index[0] * _dims[1] + index[1];
                                // 分析??下实现???辑??是当前经过column_slice之后，全部跳过部分的行×列对应信息??是内部行加载过程中的偏移量，3是内部加载的偏移??
                                uint32_t burst_offset = (std::ceil((double)(_dims[0]*_dims[1]) / MyAddressAllocator::dram_channels) * column_slice_size * column_slice_index
                                    + row_iteration_index * column_slice_size
                                    + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;
                                burst_offsets.push_back(burst_offset);
                            }
                            else {
                                if (index[0] * _dims[1] + index[1] == row_index) { // the index[0] is increased
                                    //uint32_t burst_offset = (row_iteration_index * _dims.back() + index.back()) * _precision / MyAddressAllocator::dram_burst_size;
                                    uint32_t burst_offset = (std::ceil((double)(_dims[0] * _dims[1]) / MyAddressAllocator::dram_channels) * column_slice_size * column_slice_index
                                        + row_iteration_index * column_slice_size
                                        + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;

                                    if (burst_offsets.back() != burst_offset) {
                                        burst_offsets.push_back(burst_offset);
                                    }
                                }
                                else if (index[0] * _dims[1] + index[1] < row_iteration_border) {  // the index[0] is increased and current rows are contained
                                    continue;
                                }
                                else {  // the index[0] is larger than the current row border, and the next row is contained
                                    // spdlog::info("Current row index is {}, larger than the row iteration index {}", index[0], row_iteration_index);
                                    row_iteration_index = std::floor((index[0] * _dims[1] + index[1]) / MyAddressAllocator::dram_channels);
                                    row_iteration_border = (row_iteration_index + 1) * MyAddressAllocator::dram_channels;
                                    row_index = index[0] * _dims[1] + index[1];
                                    //uint32_t burst_offset = (row_iteration_index * _dims.back() + index.back()) * _precision / MyAddressAllocator::dram_burst_size;
                                    uint32_t burst_offset = (std::ceil((double)(_dims[0] * _dims[1]) / MyAddressAllocator::dram_channels) * column_slice_size * column_slice_index
                                    + row_iteration_index * column_slice_size
                                    + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;
                                    burst_offsets.push_back(burst_offset);
                                }
                            }
                        }
                        // spdlog::info("start index={}, end index = {}, {} activation Burst offset are generated", indexes.front(), indexes.back(), burst_offsets.size());
                    }
                }
                else {
                    throw std::runtime_error("Tensor type not supported");
                }

                for (auto burst_offset : burst_offsets) {
                    addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
                    addrs.push_back(addr);
                }
            }
        }
        else if (_tensor_type == TensorType::WGT) {
            assert(_dims.size() == 2);
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
                uint32_t column_slice_size = MyAddressAllocator::weight_column_slice_size;
                uint32_t dram_burst_num = std::ceil(static_cast<double>(MyAddressAllocator::dram_burst_size)/_precision);

                uint32_t burst_times_per_col_slice = column_slice_size / dram_burst_num;
                uint32_t last_column_slice_length = _dims[1] % column_slice_size;
                uint32_t last_column_slice_burst_times = (last_column_slice_length == 0) ? burst_times_per_col_slice : std::ceil(static_cast<double>(last_column_slice_length) / dram_burst_num);

                uint32_t start_row = indexes.front()[0];
                uint32_t start_col = indexes.front()[1];
                uint32_t end_row = indexes.back()[0];
                uint32_t end_col = indexes.back()[1];

                uint32_t start_col_slice = std::floor(start_col/column_slice_size);
                uint32_t end_col_slice = std::ceil(end_col/column_slice_size);

                std::unordered_map<uint32_t, std::vector<uint32_t>> column_slice_burst_offsets;
                for (uint32_t i = start_col_slice; i <= end_col_slice; ++i) {
                    column_slice_burst_offsets[i] = std::vector<uint32_t>();
                }

                assert(column_slice_size % dram_burst_num == 0);
                for (uint32_t col_burst_index = start_col/dram_burst_num; col_burst_index <= end_col/dram_burst_num; col_burst_index++) {
                    uint32_t col_slice_id = col_burst_index / burst_times_per_col_slice;
                    column_slice_burst_offsets[col_slice_id].push_back(col_burst_index);
                }

                uint32_t column_slice_burst_offset;     // burst offset for previous column slice
                uint32_t current_slice_row_burst_times; // burst offset of current column slice
                // 下面是加载过程，按照行方向进行加载，如果遇到了column slice边界在后面换??
                for (uint32_t col_slice_id = start_col_slice; col_slice_id <= end_col_slice; ++col_slice_id) {
                    column_slice_burst_offset = col_slice_id * std::ceil(static_cast<double>(_dims[0]) / MyAddressAllocator::dram_channels) * burst_times_per_col_slice; // 前序切片??跳过offset数量
                    if (col_slice_id == end_col_slice) {
                        current_slice_row_burst_times = last_column_slice_burst_times; // 当前slice对应的是??后一个列切片，其长度小于等于column_slice
                    }
                    else {
                        current_slice_row_burst_times = burst_times_per_col_slice; // 当前slice不是??后一个列切片，其长度等于column_slice
                    }
                    for (uint32_t row_iteration_index = start_row/MyAddressAllocator::dram_channels; row_iteration_index <= end_row/MyAddressAllocator::dram_channels; ++row_iteration_index) { // 当前切片对应的行??
                        uint32_t row_iteration_offset = row_iteration_index * current_slice_row_burst_times;  // row iteration × row offset
                        for (auto col_offset:column_slice_burst_offsets[col_slice_id]) {
                            burst_offsets.push_back(column_slice_burst_offset + row_iteration_offset + col_offset);
                        }
                    }
                }
                /*
                for (int i = 0; i < indexes.size(); i++) {
                    uint32_t burst_offset = (indexes[i][0] * _dims[1] + indexes[i][1]) * _precision / MyAddressAllocator::memory_burst_size;
                    if (burst_offsets.empty()) {
                        burst_offsets.push_back(burst_offset);
                    }
                    else {
                        if (burst_offsets.back() != burst_offset) {
                            burst_offsets.push_back(burst_offset);
                        }
                    }
                }
                */
                for (auto burst_offset : burst_offsets) {
                    addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
                    addrs.push_back(addr);
                }
                // MyAddressAllocator::check_addrs(addrs);
            }
            else if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {
                uint32_t tile_n_begin = indexes[0][1];
                uint32_t tile_n_end = indexes.back()[1];
                uint32_t tile_k_begin = indexes[0][0];
                uint32_t tile_k_end = indexes.back()[0];

                if (MyAddressAllocator::IANUS_channel_parallel) { // 优先Channel并行分配
                    // 基于索引得到每一个column??在DRAM中Bank的循环分配次数，在这个计算过程中可以得到对应的bank index，以及allocate iteration
                    uint32_t ianus_column_index_begin = tile_n_begin / MyAddressAllocator::dram_channels;
                    uint32_t ianus_column_index_end = tile_n_end / MyAddressAllocator::dram_channels;
                    uint32_t ianus_column_burst_offset_begin = tile_k_begin * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t ianus_column_burst_offset_end = tile_k_end * _precision / MyAddressAllocator::dram_burst_size;

                    uint32_t ianus_column_iteration, ianus_bank_offset;
                    uint32_t row_index, column_index;

                    for (uint32_t ianus_column_offset = ianus_column_index_begin; ianus_column_offset <= ianus_column_index_end; ianus_column_offset++) {
                        ianus_column_iteration = ianus_column_offset / MyAddressAllocator::banks_per_channel;
                        ianus_bank_offset = ianus_column_offset % MyAddressAllocator::banks_per_channel;
                        for (uint32_t column_burst_offset = ianus_column_burst_offset_begin; column_burst_offset <= ianus_column_burst_offset_end; column_burst_offset++) {
                            row_index = allocate_rows[ianus_column_iteration] + column_burst_offset / MyAddressAllocator::BL_num_per_row;
                            column_index = column_burst_offset % MyAddressAllocator::BL_num_per_row;
                            auto addr = MyAddressAllocator::make_address_by_index(
                                MyAddressAllocator::rababg_bank_index[ianus_bank_offset][0],
                                MyAddressAllocator::rababg_bank_index[ianus_bank_offset][2],
                                 MyAddressAllocator::rababg_bank_index[ianus_bank_offset][1],
                                row_index, column_index,0);
                            addrs.push_back(addr);
                        }
                    }
                }
                else {
                    uint32_t ianus_column_iteration, ianus_bank_offset;
                    uint32_t channel_offset, channel_bank_offset;
                    uint32_t ianus_column_burst_offset_begin = tile_k_begin * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t ianus_column_burst_offset_end = tile_k_end * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t row_index, column_index;
                    /*
                    for (uint32_t ianus_column_index = tile_n_begin; ianus_column_index <= tile_n_end; ianus_column_index++) {
                        ianus_column_iteration = ianus_column_index / MyAddressAllocator::total_banks;
                        ianus_bank_offset = ianus_column_index % MyAddressAllocator::total_banks;
                        channel_offset = ianus_bank_offset / MyAddressAllocator::banks_per_channel;
                        channel_bank_offset = ianus_bank_offset % MyAddressAllocator::banks_per_channel;

                        for (uint32_t column_burst_offset = ianus_column_burst_offset_begin; column_burst_offset <= ianus_column_burst_offset_end; column_burst_offset++) {
                            row_index = allocate_rows[ianus_column_iteration] + column_burst_offset / MyAddressAllocator::BL_num_per_row;
                            column_index = column_burst_offset % MyAddressAllocator::BL_num_per_row;
                            auto addr = MyAddressAllocator::make_address_by_index(
                                MyAddressAllocator::rababg_bank_index[channel_bank_offset][0],
                                MyAddressAllocator::rababg_bank_index[channel_bank_offset][2],
                                 MyAddressAllocator::rababg_bank_index[channel_bank_offset][1],
                                row_index, column_index, channel_offset);
                            addrs.push_back(addr);
                        }
                    }
                    */
                    for (uint32_t column_burst_offset = ianus_column_burst_offset_begin; column_burst_offset <= ianus_column_burst_offset_end; column_burst_offset++) {
                        column_index = column_burst_offset % MyAddressAllocator::BL_num_per_row;
                        for (uint32_t ianus_column_index = tile_n_begin; ianus_column_index <= tile_n_end; ianus_column_index++) {
                            ianus_column_iteration = ianus_column_index / MyAddressAllocator::total_banks;
                            ianus_bank_offset = ianus_column_index % MyAddressAllocator::total_banks;
                            channel_offset = ianus_bank_offset / MyAddressAllocator::banks_per_channel;
                            channel_bank_offset = ianus_bank_offset % MyAddressAllocator::banks_per_channel;
                            row_index = allocate_rows[ianus_column_iteration] + column_burst_offset / MyAddressAllocator::BL_num_per_row;
                            auto addr = MyAddressAllocator::make_address_by_index(
                                MyAddressAllocator::rababg_bank_index[channel_bank_offset][0],
                                MyAddressAllocator::rababg_bank_index[channel_bank_offset][2],
                                MyAddressAllocator::rababg_bank_index[channel_bank_offset][1],
                                        row_index, column_index, channel_offset);
                            addrs.push_back(addr);
                        }
                    }
                }
                // MyAddressAllocator::check_addrs(addrs);

                /*
                assert(tile_n_begin/MyAddressAllocator::tile_width == tile_n_end/MyAddressAllocator::tile_width); // ???NPU???в???????????tile????????????????
                assert(tile_k_begin/MyAddressAllocator::tile_height == tile_k_end/MyAddressAllocator::tile_height);
                // IANUS Tile
                uint32_t IANUS_tile_index_begin = tile_n_begin / MyAddressAllocator::total_banks;
                uint32_t IANUS_tile_index_end = tile_n_end / MyAddressAllocator::total_banks;
                uint32_t IANUS_tile_burst_offset_begin = tile_k_begin * _precision / MyAddressAllocator::dram_burst_size;
                uint32_t IANUS_tile_burst_offset_end = tile_k_end * _precision / MyAddressAllocator::dram_burst_size;
                for (uint32_t load_tile_index=IANUS_tile_index_begin; load_tile_index<=IANUS_tile_index_end; load_tile_index++) {
                    for (uint32_t IANUS_tile_offset_index = IANUS_tile_burst_offset_begin; IANUS_tile_offset_index<= IANUS_tile_burst_offset_end; IANUS_tile_offset_index++) {
                        uint32_t row_index = allocate_rows[load_tile_index] + IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row;
                        //uint32_t row_index = allocate_rows[IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row];
                        uint32_t column_index = IANUS_tile_offset_index % MyAddressAllocator::BL_num_per_row;
                        for (uint32_t rank_index = 0; rank_index < MyAddressAllocator::ranks; rank_index++) {
                            for (uint32_t bank_index=0; bank_index < MyAddressAllocator::banks; bank_index++) {
                                for (uint32_t bankgroup_index=0; bankgroup_index < MyAddressAllocator::bankgroups; bankgroup_index++) {
                                    auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,0);
                                    addrs.push_back(addr);
                                }
                            }
                        }
                    }
                }

                for (uint32_t load_tile_index=IANUS_tile_index_begin; load_tile_index<=IANUS_tile_index_end; load_tile_index++) {
                    for (uint32_t rank_index = 0; rank_index < MyAddressAllocator::ranks; rank_index++) {
                        for (uint32_t bank_index=0; bank_index < MyAddressAllocator::banks; bank_index++) {
                            for (uint32_t bankgroup_index=0; bankgroup_index < MyAddressAllocator::bankgroups; bankgroup_index++) {
                                for (uint32_t IANUS_tile_offset_index = IANUS_tile_burst_offset_begin; IANUS_tile_offset_index<= IANUS_tile_burst_offset_end; IANUS_tile_offset_index++) {
                                    uint32_t row_index = allocate_rows[load_tile_index] + IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row;
                                    //uint32_t row_index = allocate_rows[IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row];
                                    uint32_t column_index = IANUS_tile_offset_index % MyAddressAllocator::BL_num_per_row;
                                    auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,0);
                                    addrs.push_back(addr);
                                }
                            }
                        }
                    }
                }

                assert(addrs.size() == MyAddressAllocator::burst_times_per_tile);
                */
                // MyAddressAllocator::check_addrs(addrs);
            }
            else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                uint32_t tile_n_begin = indexes[0][1];
                uint32_t tile_n_end = indexes.back()[1];
                uint32_t tile_k_begin = indexes[0][0];
                uint32_t tile_k_end = indexes.back()[0];

                // assert(tile_n_begin/MyAddressAllocator::tile_width == tile_n_end/MyAddressAllocator::tile_width);
                // assert(tile_k_begin/MyAddressAllocator::tile_height == tile_k_end/MyAddressAllocator::tile_height);

                uint32_t tile_index = (tile_n_begin) / MyAddressAllocator::tile_width;
                uint32_t tile_iteration = tile_index / MyAddressAllocator::allocated_tiles_per_iteration;
                uint32_t tile_offset = tile_index % MyAddressAllocator::allocated_tiles_per_iteration;
                uint32_t burst_offset_in_tile = (tile_k_begin / MyAddressAllocator::tile_height) * MyAddressAllocator::burst_times_per_tile;

                for (int i=0; i<MyAddressAllocator::burst_times_per_tile; i++) {

                    uint32_t interleave_index = burst_offset_in_tile % MyAddressAllocator::interleaved_banks_per_tile; // ???tile??weight??????????Bank
                    uint32_t column_index = burst_offset_in_tile / MyAddressAllocator::interleaved_banks_per_tile % MyAddressAllocator::BL_num_per_row;
                    uint32_t row_index = allocate_rows[tile_iteration] + burst_offset_in_tile / MyAddressAllocator::interleaved_banks_per_tile / MyAddressAllocator::BL_num_per_row;
                    uint32_t rank_index = MyAddressAllocator::interleaved_bank_index[tile_offset][interleave_index][0];
                    uint32_t bank_index = MyAddressAllocator::interleaved_bank_index[tile_offset][interleave_index][1];
                    uint32_t bankgroup_index = MyAddressAllocator::interleaved_bank_index[tile_offset][interleave_index][2];
                    auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,0);
                    addrs.push_back(addr);
                    burst_offset_in_tile++;
                }
                assert(addrs.size() == MyAddressAllocator::burst_times_per_tile);
            }
            else {
                throw std::runtime_error("Other allocation scheme not supported");
            }
            // MyAddressAllocator::check_addrs(addrs);
        }
        else if (_tensor_type == TensorType::KCache) {
            assert(_dims.size() == 2);
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {
                /* 对于KCache的地??逻辑??要完成重新的处理*/
                uint32_t tile_n_begin = indexes[0][1];
                uint32_t tile_n_end = indexes.back()[1];
                uint32_t tile_k_begin = indexes[0][0];
                uint32_t tile_k_end = indexes.back()[0];
                assert(tile_n_begin/MyAddressAllocator::tile_width == tile_n_end/MyAddressAllocator::tile_width);
                assert(tile_k_begin/MyAddressAllocator::tile_height == tile_k_end/MyAddressAllocator::tile_height);

                uint32_t begin_head_index = tile_n_begin / MyAddressAllocator::d_k;
                uint32_t end_head_index = tile_n_end / MyAddressAllocator::d_k;
                uint32_t entry_index_begin = tile_k_begin / MyAddressAllocator::kv_cache_entry_size;
                uint32_t entry_index_end = tile_k_end / MyAddressAllocator::kv_cache_entry_size;

                for (uint32_t head_index = begin_head_index; head_index <= end_head_index; head_index++) {
                    // 在n方向??要取出当前head??在范围的index
                    uint32_t current_head_n_begin = std::max(head_index * MyAddressAllocator::d_k, tile_n_begin);
                    uint32_t current_head_n_end = std::min((head_index+1) * MyAddressAllocator::d_k - 1, tile_n_end);

                    uint32_t head_cache_burst_offset_begin = current_head_n_begin % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t head_cache_offset_end = current_head_n_end % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t head_cache_burst_times = MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;

                    uint32_t column_index = 0;
                    uint32_t total_bank_index;
                    uint32_t row_iteration;
                    uint32_t row_offset;
                    uint32_t channel_index;
                    uint32_t bank_offset;

                    // 首先??要基于head和entry的index与head完成定位，确定在哪个channel与bank中，之后基于offset确定在哪个row??
                    for (uint32_t entry_index = entry_index_begin; entry_index<= entry_index_end; entry_index++) {
                        total_bank_index = entry_index * MyAddressAllocator::h_kv + head_index;  // 总的分配的行??
                        row_iteration = total_bank_index / MyAddressAllocator::total_banks;
                        row_offset = total_bank_index % MyAddressAllocator::total_banks;

                        channel_index = row_offset % MyAddressAllocator::dram_channels;
                        bank_offset = (row_offset / MyAddressAllocator::dram_channels) % MyAddressAllocator::banks_per_channel;

                        for (uint32_t token_offset = 0; token_offset< MyAddressAllocator::kv_cache_entry_size; token_offset++) {
                            for (uint32_t burst_offset = head_cache_burst_offset_begin; burst_offset <= head_cache_offset_end; burst_offset++) {
                                column_index = token_offset * MyAddressAllocator::burst_times_per_KCache_row + burst_offset;
                                auto addr = MyAddressAllocator::make_address_by_index(
                                    MyAddressAllocator::rababg_bank_index[bank_offset][0],
                                    MyAddressAllocator::rababg_bank_index[bank_offset][1],
                                    MyAddressAllocator::rababg_bank_index[bank_offset][2],
                                    kv_cache_allocate_rows[0][row_iteration],
                                    column_index,
                                    channel_index
                                    );
                                addrs.push_back(addr);
                            }
                        }
                    }
                }


                // 以下是之前的写法
                /*
                uint32_t head_index = tile_n_begin / MyAddressAllocator::d_k; // tile_n_begin
                uint32_t head_cache_burst_offset_begin = tile_n_begin % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                uint32_t head_cache_offset_end = tile_n_end % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                uint32_t head_cache_burst_times = MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;

                uint32_t column_index = 0;
                uint32_t total_bank_index;
                uint32_t row_iteration;
                uint32_t row_offset;
                uint32_t channel_index;
                uint32_t bank_offset;

                // 首先??要基于head和entry的index与head完成定位，确定在哪个channel与bank中，之后基于offset确定在哪个row??
                for (uint32_t entry_index = entry_index_begin; entry_index<= entry_index_end; entry_index++) {
                    total_bank_index = entry_index * MyAddressAllocator::h_kv + head_index;  // 总的分配的行??
                    row_iteration = total_bank_index / MyAddressAllocator::total_banks;
                    row_offset = total_bank_index % MyAddressAllocator::total_banks;

                    channel_index = row_offset % MyAddressAllocator::dram_channels;
                    bank_offset = (row_offset / MyAddressAllocator::dram_channels) % MyAddressAllocator::banks_per_channel;

                    for (uint32_t token_offset = 0; token_offset< MyAddressAllocator::kv_cache_entry_size; token_offset++) {
                        for (uint32_t burst_offset = head_cache_burst_offset_begin; burst_offset <= head_cache_offset_end; burst_offset++) {
                            column_index = token_offset * MyAddressAllocator::burst_times_per_KCache_row + burst_offset;
                            auto addr = MyAddressAllocator::make_address_by_index(
                                MyAddressAllocator::rababg_bank_index[bank_offset][0],
                                MyAddressAllocator::rababg_bank_index[bank_offset][1],
                                MyAddressAllocator::rababg_bank_index[bank_offset][2],
                                kv_cache_allocate_rows[0][row_iteration],
                                column_index,
                                channel_index
                                );
                            addrs.push_back(addr);
                        }
                    }
                }
                */

                // spdlog::info("{} address is generate for Head index {}, load KCache token index {} - {}, token offset {} - {}",
                //    addrs.size(), head_index, tile_k_begin, tile_k_end, tile_n_begin, tile_n_end);
                // MyAddressAllocator::check_addrs(addrs);

                /*
                uint32_t IANUS_tile_index_begin = tile_n_begin / MyAddressAllocator::total_banks;
                uint32_t IANUS_tile_index_end = tile_n_end / MyAddressAllocator::total_banks;
                uint32_t IANUS_tile_burst_offset_begin = tile_k_begin * _precision / MyAddressAllocator::dram_burst_size;
                uint32_t IANUS_tile_burst_offset_end = tile_k_end * _precision / MyAddressAllocator::dram_burst_size;

                for (uint32_t load_tile_index=IANUS_tile_index_begin; load_tile_index<=IANUS_tile_index_end; load_tile_index++) {
                    for (uint32_t rank_index = 0; rank_index < MyAddressAllocator::ranks; rank_index++) {
                        for (uint32_t bank_index=0; bank_index < MyAddressAllocator::banks; bank_index++) {
                            for (uint32_t bankgroup_index=0; bankgroup_index < MyAddressAllocator::bankgroups; bankgroup_index++) {
                                for (uint32_t IANUS_tile_offset_index = IANUS_tile_burst_offset_begin; IANUS_tile_offset_index<= IANUS_tile_burst_offset_end; IANUS_tile_offset_index++) {
                                    uint32_t row_index = kv_cache_allocate_rows[head_index][load_tile_index] + IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row;
                                    //uint32_t row_index = allocate_rows[IANUS_tile_offset_index / MyAddressAllocator::BL_num_per_row];
                                    uint32_t column_index = IANUS_tile_offset_index % MyAddressAllocator::BL_num_per_row;
                                    auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,0);
                                    addrs.push_back(addr);
                                }
                            }
                        }
                    }
                }
                */
            }
            else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                // Generate KCache arrangement based on the index
                uint32_t tile_n_begin = indexes[0][1];  // start K Column, compute current tile
                uint32_t tile_n_end = indexes.back()[1]; // end K Column, compute current tile
                uint32_t tile_m_begin = indexes[0][0]; // start N Row
                uint32_t tile_m_end = indexes.back()[0]; // end N Row

                uint32_t begin_head_index = tile_n_begin / MyAddressAllocator::d_k;
                uint32_t end_head_index = tile_n_end / MyAddressAllocator::d_k;

                for (uint32_t head_index = begin_head_index; head_index <= end_head_index; head_index++) {
                    uint32_t current_head_n_begin = std::max(head_index * MyAddressAllocator::d_k, tile_n_begin);
                    uint32_t current_head_n_end = std::min((head_index+1) * MyAddressAllocator::d_k - 1, tile_n_end);

                    // Get the allocated row and corresponding bank index based on head index
                    uint32_t allocate_channel_index = head_index % MyAddressAllocator::dram_channels;
                    uint32_t channel_head_iteration_index = head_index % MyAddressAllocator::allocated_KCache_head_per_iteration / MyAddressAllocator::dram_channels;

                    // Get bank-row-column index based on KCache row index, require the offset
                    uint32_t KCache_row_allocate_begin = tile_m_begin;
                    uint32_t KCache_row_allocate_end = tile_m_end;

                    uint32_t KCache_row_burst_offset_begin = current_head_n_begin % MyAddressAllocator::d_k / (MyAddressAllocator::dram_burst_size/_precision);
                    uint32_t KCache_row_burst_offset_end = current_head_n_end % MyAddressAllocator::d_k / (MyAddressAllocator::dram_burst_size/_precision);

                    for (uint32_t KCache_row_allocate = KCache_row_allocate_begin; KCache_row_allocate <= KCache_row_allocate_end; KCache_row_allocate++) {
                        uint32_t head_row_index = KCache_row_allocate / MyAddressAllocator::KCache_rows_per_allocation; // current KCache row and corresponding Bank and Row index
                        uint32_t row_index = kv_cache_allocate_rows[head_index][head_row_index];
                        uint32_t head_row_offset = (KCache_row_allocate % MyAddressAllocator::KCache_rows_per_allocation) / MyAddressAllocator::KCache_interleaved_banks_per_head;

                        uint32_t interleaved_bank_index = KCache_row_allocate % MyAddressAllocator::KCache_interleaved_banks_per_head;
                        uint32_t rank_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][0];
                        uint32_t bank_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][1];
                        uint32_t bankgroup_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][2];

                        uint32_t KCache_burst_offset = head_row_offset * MyAddressAllocator::burst_times_per_KCache_row;
                        for (uint32_t KCache_row_burst_offset = KCache_row_burst_offset_begin; KCache_row_burst_offset <= KCache_row_burst_offset_end; KCache_row_burst_offset++) {
                            uint32_t column_index = KCache_burst_offset + KCache_row_burst_offset;
                            auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index, allocate_channel_index);
                            addrs.push_back(addr);
                        }
                    }
                }

                //
                /*
                // spdlog::info("Current K Cache index is n = {}-{}, k = {}-{}", indexes.front()[0], indexes.back()[0], indexes.front()[1], indexes.back()[1]);
                // Calculate the storage location corresponding to the current attention head based on tile_k_begin
                uint32_t head_index = tile_n_begin / MyAddressAllocator::d_k; // get head index based on tile_n_begin
                assert(head_index == tile_n_end / MyAddressAllocator::d_k);  // make sure all the accessed data are in the same head

                // Get the allocated row and corresponding bank index based on head index
                uint32_t allocate_channel_index = head_index % MyAddressAllocator::dram_channels;
                uint32_t channel_head_iteration_index = head_index % MyAddressAllocator::allocated_KCache_head_per_iteration / MyAddressAllocator::dram_channels;

                // Get bank-row-column index based on KCache row index, require the offset
                uint32_t KCache_row_allocate_begin = tile_m_begin;
                uint32_t KCache_row_allocate_end = tile_m_end;

                uint32_t KCache_row_burst_offset_begin = tile_n_begin % MyAddressAllocator::d_k / (MyAddressAllocator::dram_burst_size/_precision);
                uint32_t KCache_row_burst_offset_end = tile_n_end % MyAddressAllocator::d_k / (MyAddressAllocator::dram_burst_size/_precision);

                for (uint32_t KCache_row_allocate = KCache_row_allocate_begin; KCache_row_allocate <= KCache_row_allocate_end; KCache_row_allocate++) {
                    uint32_t head_row_index = KCache_row_allocate / MyAddressAllocator::KCache_rows_per_allocation; // current KCache row and corresponding Bank and Row index
                    uint32_t row_index = kv_cache_allocate_rows[head_index][head_row_index];
                    uint32_t head_row_offset = (KCache_row_allocate % MyAddressAllocator::KCache_rows_per_allocation) / MyAddressAllocator::KCache_interleaved_banks_per_head;

                    uint32_t interleaved_bank_index = KCache_row_allocate % MyAddressAllocator::KCache_interleaved_banks_per_head;
                    uint32_t rank_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][0];
                    uint32_t bank_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][1];
                    uint32_t bankgroup_index = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][2];

                    uint32_t KCache_burst_offset = head_row_offset * MyAddressAllocator::burst_times_per_KCache_row;
                    for (uint32_t KCache_row_burst_offset = KCache_row_burst_offset_begin; KCache_row_burst_offset <= KCache_row_burst_offset_end; KCache_row_burst_offset++) {
                        uint32_t column_index = KCache_burst_offset + KCache_row_burst_offset;
                        auto addr = MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index, allocate_channel_index);
                        addrs.push_back(addr);
                    }
                }
                */
                // spdlog::info("{} burst addrs for one tile of K Cache are generated", addrs.size());
                // MyAddressAllocator::check_addrs(addrs);
            }
        }
        else if (_tensor_type == TensorType::VCache) {
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU or MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {
                uint32_t tile_n_begin = indexes[0][1];
                uint32_t tile_n_end = indexes.back()[1];
                uint32_t tile_k_begin = indexes[0][0];
                uint32_t tile_k_end = indexes.back()[0];
                assert(tile_n_begin/MyAddressAllocator::tile_width == tile_n_end/MyAddressAllocator::tile_width);
                assert(tile_k_begin/MyAddressAllocator::tile_height == tile_k_end/MyAddressAllocator::tile_height);

                uint32_t begin_head_index = tile_n_begin / MyAddressAllocator::d_k;
                uint32_t end_head_index = tile_n_end / MyAddressAllocator::d_k;
                uint32_t entry_index_begin = tile_k_begin / MyAddressAllocator::kv_cache_entry_size;
                uint32_t entry_index_end = tile_k_end / MyAddressAllocator::kv_cache_entry_size;

                for (uint32_t head_index = begin_head_index; head_index <= end_head_index; head_index++) {
                    uint32_t current_head_n_begin = std::max(head_index * MyAddressAllocator::d_k, tile_n_begin);
                    uint32_t current_head_n_end = std::min((head_index+1) * MyAddressAllocator::d_k - 1, tile_n_end);

                    uint32_t head_cache_burst_offset_begin = current_head_n_begin % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t head_cache_offset_end = current_head_n_end % MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;
                    uint32_t head_cache_burst_times = MyAddressAllocator::d_k * _precision / MyAddressAllocator::dram_burst_size;

                    uint32_t column_index = 0;
                    uint32_t total_bank_index;
                    uint32_t row_iteration;
                    uint32_t row_offset;
                    uint32_t channel_index;
                    uint32_t bank_offset;

                    for (uint32_t entry_index = entry_index_begin; entry_index<= entry_index_end; entry_index++) {
                        total_bank_index = entry_index * MyAddressAllocator::h_kv + head_index;
                        row_iteration = total_bank_index / MyAddressAllocator::total_banks;
                        row_offset = total_bank_index % MyAddressAllocator::total_banks;

                        channel_index = row_offset % MyAddressAllocator::dram_channels;
                        bank_offset = (row_offset / MyAddressAllocator::dram_channels) % MyAddressAllocator::banks_per_channel;

                        for (uint32_t token_offset = 0; token_offset< MyAddressAllocator::kv_cache_entry_size; token_offset++) {
                            for (uint32_t burst_offset = head_cache_burst_offset_begin; burst_offset <= head_cache_offset_end; burst_offset++) {
                                column_index = token_offset * MyAddressAllocator::burst_times_per_KCache_row + burst_offset;
                                auto addr = MyAddressAllocator::make_address_by_index(
                                    MyAddressAllocator::rababg_bank_index[bank_offset][0],
                                    MyAddressAllocator::rababg_bank_index[bank_offset][1],
                                    MyAddressAllocator::rababg_bank_index[bank_offset][2],
                                    kv_cache_allocate_rows[0][row_iteration],
                                    column_index,
                                    channel_index
                                    );
                                addrs.push_back(addr);
                            }
                        }
                    }
                }
            }
            else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                assert(_dims.size() == 2);
                // Generate corresponding addresses according to the proposed KCache layout scheme, and complete calculations based on index information
                uint32_t tile_k_begin = indexes[0][0];   // begin column
                uint32_t tile_k_end = indexes.back()[0]; // end column
                uint32_t tile_n_begin = indexes[0][1];   // start column, get the corresponding tile
                uint32_t tile_n_end = indexes.back()[1]; // end column, get the corresponding tile

                uint32_t begin_head_index = tile_n_begin / MyAddressAllocator::d_k;
                uint32_t end_head_index = tile_n_end / MyAddressAllocator::d_k;

                for (uint32_t head_index = begin_head_index; head_index <= end_head_index; head_index++) {
                    uint32_t current_head_n_begin = std::max(head_index * MyAddressAllocator::d_k, tile_n_begin);
                    uint32_t current_head_n_end = std::min((head_index+1) * MyAddressAllocator::d_k - 1, tile_n_end);

                    uint32_t allocate_channel_index = head_index % MyAddressAllocator::dram_channels;
                    uint32_t channel_head_iteration_index = head_index % MyAddressAllocator::allocated_VCache_head_per_iteration / MyAddressAllocator::dram_channels;

                    // compute the row allocate allocation based on tile index
                    uint32_t VCache_row_allocate_iteration_begin = tile_k_begin / MyAddressAllocator::VCache_burst_row_unit;
                    uint32_t VCache_row_allocate_iteration_end = tile_k_end / MyAddressAllocator::VCache_burst_row_unit;
                    uint32_t VCache_column_iteration_begin = current_head_n_begin % MyAddressAllocator::d_k;
                    uint32_t VCache_column_iteration_end = current_head_n_end % MyAddressAllocator::d_k;

                    for (uint32_t VCache_row_allocate_iteration = VCache_row_allocate_iteration_begin; VCache_row_allocate_iteration <= VCache_row_allocate_iteration_end; VCache_row_allocate_iteration++) {
                        uint32_t row_index = kv_cache_allocate_rows[head_index][VCache_row_allocate_iteration / MyAddressAllocator::VCache_row_units_per_allocation];
                        for (uint32_t VCache_column_iteration = VCache_column_iteration_begin; VCache_column_iteration <= VCache_column_iteration_end; VCache_column_iteration++) {
                            uint32_t interleave_offset = VCache_column_iteration / MyAddressAllocator::VCache_interleaved_banks_per_head;
                            uint32_t column_index = (VCache_row_allocate_iteration % MyAddressAllocator::VCache_row_units_per_allocation) * MyAddressAllocator::VCache_columns_per_bank + interleave_offset;
                            uint32_t interleaved_bank_index = VCache_column_iteration % MyAddressAllocator::VCache_interleaved_banks_per_head;
                            uint32_t rank_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][0];
                            uint32_t bank_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][1];
                            uint32_t bankgroup_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][2];
                            addrs.push_back(MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,allocate_channel_index));
                        }
                    }
                }

                /*
                uint32_t head_index = tile_n_begin / MyAddressAllocator::d_k; // get head index
                assert(head_index == tile_n_end / MyAddressAllocator::d_k); // make sure all the operation is the same Attention head
                // check the corresponding channel and bank index based on the head index
                uint32_t allocate_channel_index = head_index % MyAddressAllocator::dram_channels;
                uint32_t channel_head_iteration_index = head_index % MyAddressAllocator::allocated_VCache_head_per_iteration / MyAddressAllocator::dram_channels;

                // compute the row allocate allocation based on tile index
                uint32_t VCache_row_allocate_iteration_begin = tile_k_begin / MyAddressAllocator::VCache_burst_row_unit;
                uint32_t VCache_row_allocate_iteration_end = tile_k_end / MyAddressAllocator::VCache_burst_row_unit;
                uint32_t VCache_column_iteration_begin = tile_n_begin % MyAddressAllocator::d_k;
                uint32_t VCache_column_iteration_end = tile_n_end % MyAddressAllocator::d_k;

                for (uint32_t VCache_row_allocate_iteration = VCache_row_allocate_iteration_begin; VCache_row_allocate_iteration <= VCache_row_allocate_iteration_end; VCache_row_allocate_iteration++) {
                    uint32_t row_index = kv_cache_allocate_rows[head_index][VCache_row_allocate_iteration / MyAddressAllocator::VCache_row_units_per_allocation];
                    for (uint32_t VCache_column_iteration = VCache_column_iteration_begin; VCache_column_iteration <= VCache_column_iteration_end; VCache_column_iteration++) {
                        uint32_t interleave_offset = VCache_column_iteration / MyAddressAllocator::VCache_interleaved_banks_per_head;
                        uint32_t column_index = (VCache_row_allocate_iteration % MyAddressAllocator::VCache_row_units_per_allocation) * MyAddressAllocator::VCache_columns_per_bank + interleave_offset;
                        uint32_t interleaved_bank_index = VCache_column_iteration % MyAddressAllocator::VCache_interleaved_banks_per_head;
                        uint32_t rank_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][0];
                        uint32_t bank_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][1];
                        uint32_t bankgroup_index = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration_index][interleaved_bank_index][2];
                        addrs.push_back(MyAddressAllocator::make_address_by_index(rank_index, bankgroup_index, bank_index, row_index, column_index,allocate_channel_index));
                    }
                }
                */
                // spdlog::info("{} burst addrs for one tile of V Cache are generated", addrs.size());
                // MyAddressAllocator::check_addrs(addrs);
            }
        }
        else {
            throw std::runtime_error("Unsupported Tensor Type");
        }
    }
    return addrs;
}


std::vector<addr_type> MyTensor::generate_addrs_based_on_indexes(std::vector<std::vector<uint32_t>> indexes,
                                                                 AllocationScheme layout_scheme) {
    ScopedAllocationScheme scoped_scheme(layout_scheme);
    return generate_addrs_based_on_indexes(std::move(indexes));
}

std::vector<addr_type> MyTensor::generate_addrs_based_on_indexes_attention(std::vector<std::vector<uint32_t>> indexes, uint32_t head_index) {
    if (indexes.size() == 0) {
        return {};
    }
    assert(_tensor_type == TensorType::ACT);
    // dim = 3, current tensor is QKT or S
    std::vector<addr_type> addrs;
    std::vector<uint32_t> burst_offsets;

    if (_dims[_dims.size() - 2] == 1) {
        uint32_t head_burst_offset = head_index * _dims[_dims.size() - 2] * _dims[_dims.size() - 1] * _precision / MyAddressAllocator::memory_burst_size;
        for (auto index : indexes) {
            uint32_t burst_offset = head_burst_offset + (index[0] * _dims[1] + index[1]) * _precision / MyAddressAllocator::memory_burst_size;
            if (burst_offsets.empty()) {
                burst_offsets.push_back(burst_offset);
            }
            else {
                if (burst_offsets.back() != burst_offset) {
                    burst_offsets.push_back(burst_offset);
                }
            }
        }
    }
    else {
        uint32_t column_slice_size = MyAddressAllocator::act_column_slice_size;// Config::system_config.model_n_embd;
        uint32_t column_slice_index = 0;
        uint32_t column_slice_offset = 0;

        uint32_t row_iteration_index = 0;
        uint32_t row_iteration_border = 0;
        uint32_t row_index = 0;

        uint32_t head_row_offset = head_index * std::ceil(static_cast<double>(_dims[_dims.size() - 2]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::dram_channels;

        for (auto index : indexes) {
            column_slice_index = std::floor(index[1]/column_slice_size);
            column_slice_offset = index[1] % column_slice_size;

            if (burst_offsets.empty()) {
                // The first burst must be available, and initial the
                row_iteration_index = std::floor((head_row_offset + index[0]) / MyAddressAllocator::dram_channels);
                row_iteration_border = (row_iteration_index + 1) * MyAddressAllocator::dram_channels;
                row_index = head_row_offset + index[0];
                uint32_t burst_offset = (std::ceil(static_cast<double>(_dims[_dims.size() - 2]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::h * column_slice_size * column_slice_index // multi-head row offset
                    + row_iteration_index * column_slice_size
                    + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;
                burst_offsets.push_back(burst_offset);
            }
            else {
                if (head_row_offset + index[0] == row_index) { // the index[0] is increased
                    uint32_t burst_offset = (std::ceil(static_cast<double>(_dims[_dims.size() - 2]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::h * column_slice_size * column_slice_index // multi-head row offset
                        + row_iteration_index * column_slice_size
                        + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;
                    if (burst_offsets.back() != burst_offset) {
                        burst_offsets.push_back(burst_offset);
                    }
                }
                else if (head_row_offset + index[0] < row_iteration_border) {
                    continue;
                }
                else { // the index[0] is larger than the current row border, and the next row is contained and update current border
                    row_iteration_index = std::floor((head_row_offset + index[0]) / MyAddressAllocator::dram_channels);
                    row_iteration_border = (row_iteration_index + 1) * MyAddressAllocator::dram_channels;
                    row_index = head_row_offset + index[0];

                    uint32_t burst_offset = (std::ceil(static_cast<double>(_dims[_dims.size() - 2]) / MyAddressAllocator::dram_channels) * MyAddressAllocator::h  // multi-head row offset
                                        * column_slice_size * column_slice_index
                                        + row_iteration_index * column_slice_size
                                        + column_slice_offset) * _precision / MyAddressAllocator::dram_burst_size;
                    burst_offsets.push_back(burst_offset);
                }
            }
        }
    }
    for (auto burst_offset : burst_offsets) {
        addr_type addr = MyAddressAllocator::get_sequence_address(burst_offset, start_outer_row_loop, start_middle_row_loop, start_inner_row_loop, start_row, start_col);
        addrs.push_back(addr);
    }
    // spdlog::info("{} addr offset are generated for head {}, begin index {}, end index{}", burst_offsets.size(), head_index, indexes.front(), indexes.back());
    return addrs;
}


std::vector<addr_type> MyTensor::generate_pim_comp_addrs(std::vector<uint32_t> start_rows, std::vector<uint32_t> indexes) {

    std::vector<addr_type> pim_comp_addrs;
    std::vector<uint32_t> burst_offsets;

    for (int i = 0; i < indexes.size(); i++) {
        uint32_t burst_offset = indexes[i] * _precision / MyAddressAllocator::dram_burst_size;
        if (burst_offsets.empty()) {
            burst_offsets.push_back(burst_offset);
        }
        else {
            if (burst_offsets.back() != burst_offset ) {
                burst_offsets.push_back(burst_offset);
            }
        }
    }

    assert(_tensor_type == TensorType::WGT);
    uint32_t weight_interleave_column_number = MyAddressAllocator::bank_allocated_columns_per_tile;
    for (auto tile_begin_row : start_rows) {
        for (uint32_t i=0; i<burst_offsets.size(); i++) {
            uint32_t burst_offset = burst_offsets[i] * weight_interleave_column_number;
            for (uint32_t j=0; j<weight_interleave_column_number; j++) {
                auto addr = MyAddressAllocator::get_sequence_address_pim(burst_offset+j, tile_begin_row, 0);
                pim_comp_addrs.push_back(addr);
            }
        }
    }
    return pim_comp_addrs;
}



std::vector<addr_type> MyTensor::generate_pim_comp_addrs_attention(uint32_t head_iteration_index, std::vector<std::vector<uint32_t>> indexes) {

    std::vector<addr_type> pim_comp_addrs;
    /*
    if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU) {
        throw std::runtime_error("Unsupported Now");
    }
    else if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
    */
    if (_tensor_type == TensorType::KCache) {
        auto row_indexes = kv_cache_allocate_rows[head_iteration_index * MyAddressAllocator::allocated_KCache_head_per_iteration];

        // The start row index and burst offset can get based on the Head index
        std::vector<uint32_t> burst_offsets;
        uint32_t KCache_k_begin = indexes[0][0];
        uint32_t KCache_k_end = indexes.back()[0];
        uint32_t KCache_n_begin = indexes[0][1];
        uint32_t KCache_n_end = indexes.back()[1];

        // burst offset for each KCache row can get based on the index information
        uint32_t Kcache_row_burst_offset_begin = std::ceil(static_cast<double>(KCache_k_begin) * _precision /MyAddressAllocator::dram_burst_size);
        uint32_t Kcache_row_burst_offset_end = std::ceil(static_cast<double>(KCache_k_end) * _precision /MyAddressAllocator::dram_burst_size);
        uint32_t k_cache_row_allocate_iteration_begin = std::ceil(static_cast<double>(KCache_n_begin)/MyAddressAllocator::KCache_interleaved_banks_per_head);
        uint32_t k_cache_row_allocate_iteration_end = std::ceil(static_cast<double>(KCache_n_end)/MyAddressAllocator::KCache_interleaved_banks_per_head);

        for (uint32_t row_offset = k_cache_row_allocate_iteration_begin; row_offset < k_cache_row_allocate_iteration_end; row_offset++) {
            for (uint32_t row_burst_offset = Kcache_row_burst_offset_begin; row_burst_offset < Kcache_row_burst_offset_end; row_burst_offset++) {
                burst_offsets.push_back( row_offset * MyAddressAllocator::burst_times_per_KCache_row + row_burst_offset);
            }
        }
        // Corresponding addr based on index
        for (auto burst_offset : burst_offsets) {
            pim_comp_addrs.push_back(MyAddressAllocator::get_sequence_address_pim(burst_offset, row_indexes, 0));
        }
        // spdlog::info("{} PIM Comp Addrs has been generated", pim_comp_addrs.size());
        // MyAddressAllocator::check_addrs(pim_comp_addrs);
    }
    else if (_tensor_type == TensorType::VCache) {
        auto row_indexes = kv_cache_allocate_rows[head_iteration_index * MyAddressAllocator::allocated_KCache_head_per_iteration];

        std::vector<uint32_t> burst_offsets;
        uint32_t VCache_k_begin = indexes[0][0];
        uint32_t VCache_k_end = indexes.back()[0];
        uint32_t VCache_n_begin = indexes[0][1];
        uint32_t VCache_n_end = indexes.back()[1];

        uint32_t VCache_k_burst_offset_begin = std::ceil(static_cast<double>(VCache_k_begin) * _precision /MyAddressAllocator::dram_burst_size);
        uint32_t VCache_k_burst_offset_end = std::ceil(static_cast<double>(VCache_k_end) * _precision /MyAddressAllocator::dram_burst_size);
        uint32_t VCache_n_interleave_begin = std::ceil(static_cast<double>(VCache_n_begin) / MyAddressAllocator::VCache_interleaved_banks_per_head);
        uint32_t VCache_n_interleave_end = std::ceil(static_cast<double>(VCache_n_end) / MyAddressAllocator::VCache_interleaved_banks_per_head);

        for (uint32_t VCache_k_burst_offset = VCache_k_burst_offset_begin; VCache_k_burst_offset < VCache_k_burst_offset_end; VCache_k_burst_offset++) {
            for (uint32_t VCache_n_interleave_offset = VCache_n_interleave_begin; VCache_n_interleave_offset < VCache_n_interleave_end; VCache_n_interleave_offset++) {
                burst_offsets.push_back(VCache_k_burst_offset * MyAddressAllocator::VCache_columns_per_bank + VCache_n_interleave_offset);
            }
        }
        for (auto burst_offset : burst_offsets) {
            pim_comp_addrs.push_back(MyAddressAllocator::get_sequence_address_pim(burst_offset, row_indexes, 0));
        }
        // MyAddressAllocator::check_addrs(pim_comp_addrs);
        // spdlog::info("{} PIM Comp Addrs has been generated", pim_comp_addrs.size());
    }

    return pim_comp_addrs;
}

/*
std::vector<addr_type> MyTensor::generate_pim_addrs_based_on_vector_indexes(std::vector<uint32_t> indexes) {
    // TODO:: Incomplete and Not Used
    // 1-D Vector
    std::vector<addr_type> addrs;
    std::vector<uint32_t> burst_offsets;
    for (int i = 0; i < indexes.size(); i++) {
        uint32_t burst_offset = indexes[i] * _precision / MyAddressAllocator::dram_burst_size;
        // generate burst offset based on bias index
        if (burst_offsets.empty()) {
            burst_offsets.push_back(burst_offset);
        }
        else {
            if (burst_offsets.back() != burst_offset ) {
                burst_offsets.push_back(burst_offset);
            }
        }
    }

    if (_tensor_type == TensorType::WGT) {
        uint32_t weight_interleave_column_number = MyAddressAllocator::bank_allocated_columns_per_tile;
        for (auto tile_begin_row : allocate_rows) {
            uint32_t burst_offset = burst_offsets.front() * weight_interleave_column_number;
            for (uint32_t i=0; i<burst_offsets.size(); i++) {
                for (uint32_t j=0; j<weight_interleave_column_number; j++) {
                    auto addr = MyAddressAllocator::get_sequence_address_pim(burst_offset, tile_begin_row, 0);
                    addrs.push_back(addr);
                    burst_offset++;
                }
            }
        }
        // assert(addrs.size() == allocate_rows.size() * burst_offsets.size() * MyAddressAllocator::bank_allocated_columns_per_tile);
        // spdlog::info("{} addrs is generated for {} times burst for weight matrix", addrs.size(), allocate_rows.size() * burst_offsets.size() * MyAddressAllocator::bank_allocated_columns_per_tile);
    }
    else if (_tensor_type == TensorType::KCache) {

    }
    else if (_tensor_type == TensorType::VCache) {

    }
    else {
        throw std::runtime_error("unsupported PIM tensor type");
    }
    return addrs;
}


std::vector<addr_type> MyTensor::generate_pim_addrs_based_on_kcache_indexes(uint32_t head_tile_index, std::vector<uint32_t> indexes) {
    // TODO:: Incomplete and Not Used
    std::vector<addr_type> addrs;

    if (_tensor_type == TensorType::KCache) {
        uint32_t allocated_KCache_row_per_bank = std::ceil((double)_dims[0] / MyAddressAllocator::KCache_interleaved_banks_per_head / MyAddressAllocator::dram_channels); // ????????KCache Head???Bank??????
        std::vector<uint32_t> kcache_row_burst_offsets;
        for (int i = 0; i < indexes.size(); i++) {
            uint32_t burst_offset = indexes[i] * _precision / MyAddressAllocator::dram_burst_size;
            if (kcache_row_burst_offsets.empty()) {
                kcache_row_burst_offsets.push_back(burst_offset);
            }
            else {
                if (kcache_row_burst_offsets.back() != burst_offset) {
                    kcache_row_burst_offsets.push_back(burst_offset);
                }
            }
        }

        std::vector<uint32_t> burst_offsets;
        for (uint32_t kcache_row_index=0; kcache_row_index < allocated_KCache_row_per_bank; kcache_row_index++) {
            for (auto row_burst_offset: kcache_row_burst_offsets) {
                burst_offsets.push_back(kcache_row_index * MyAddressAllocator::burst_times_per_KCache_row + row_burst_offset);
            }
        }

        for (uint32_t burst_offset: burst_offsets) {
            auto addr = MyAddressAllocator::get_sequence_address_pim(burst_offset, kv_cache_allocate_rows[head_tile_index * MyAddressAllocator::allocated_KCache_head_per_iteration], 0);
            addrs.push_back(addr);
        }
        // MyAddressAllocator::check_addrs(addrs);
    }
    else if (_tensor_type == TensorType::VCache) {
        uint32_t allocated_vcache_column_per_bank = MyAddressAllocator::VCache_columns_per_bank;
        uint32_t VCache_interleave_rows = MyAddressAllocator::VCache_burst_row_unit;
        assert(VCache_interleave_rows ==  MyAddressAllocator::dram_burst_size/_precision);
        std::vector<uint32_t> vcache_column_burst_offsets;
        for (int i = 0; i < indexes.size(); i++) {
            uint32_t burst_offset = indexes[i] / VCache_interleave_rows;
            if (vcache_column_burst_offsets.empty()) {
                vcache_column_burst_offsets.push_back(burst_offset);
            }
            else {
                if (vcache_column_burst_offsets.back() != burst_offset) {
                    vcache_column_burst_offsets.push_back(burst_offset);
                }
            }
        }

        // burst offset:
        uint32_t burst_offset = vcache_column_burst_offsets.front() * allocated_vcache_column_per_bank;

        for (uint32_t i=0; i<vcache_column_burst_offsets.size(); i++) {
            for (uint32_t j=0; j<allocated_vcache_column_per_bank; j++) {
                // auto addr = MyAddressAllocator::get_sequence_address_pim(burst_offset, tile_begin_row, 0);
                auto addr = MyAddressAllocator::get_sequence_address_pim(burst_offset, kv_cache_allocate_rows[head_tile_index * MyAddressAllocator::allocated_KCache_head_per_iteration], 0);
                addrs.push_back(addr);
                burst_offset++;
            }
        }
        // MyAddressAllocator::check_addrs(addrs);
    }
    return addrs;
}
*/

std::vector<addr_type> MyTensor::generate_weight_column_major_1kb_base_addrs(uint32_t k0, uint32_t n0, uint32_t k1, uint32_t n1, uint32_t granularity_bytes) {
    if (_tensor_type != TensorType::WGT) {
        throw std::runtime_error("generate_weight_column_major_1kb_base_addrs: WGT only");
    }
    if (_dims.size() != 2) {
        throw std::runtime_error("generate_weight_column_major_1kb_base_addrs: 2D only");
    }
    if (granularity_bytes == 0) {
        granularity_bytes = 1024;
    }
    if (granularity_bytes % _precision != 0) {
        throw std::runtime_error("granularity_bytes must be a multiple of weight precision (e.g. 1024 for FP16 512 elements)");
    }

    const uint32_t K = _dims[0];
    const uint32_t N = _dims[1];

    k0 = std::min(k0, K);
    k1 = std::min(std::max(k1, k0), K);
    n0 = std::min(n0, N);
    n1 = std::min(std::max(n1, n0), N);
    if (k1 <= k0 || n1 <= n0) {
        return {};
    }

    const uint64_t region_byte_min = (static_cast<uint64_t>(n0) * K + k0) * _precision;
    const uint64_t region_byte_end_excl = (static_cast<uint64_t>(n1 - 1) * K + (k1 - 1) + 1) * _precision;

    uint64_t first_chunk = region_byte_min / granularity_bytes * granularity_bytes;
    if (first_chunk < region_byte_min) {
        first_chunk += granularity_bytes;
    }

    const uint64_t mbs = MyAddressAllocator::memory_burst_size;
    if (mbs == 0) {
        throw std::runtime_error("memory_burst_size is zero");
    }

    std::vector<addr_type> addrs;
    uint64_t prev_burst = UINT64_MAX;
    for (uint64_t b = first_chunk; b < region_byte_end_excl; b += granularity_bytes) {
        const uint64_t burst_offset = b / mbs;
        if (burst_offset == prev_burst) {
            continue;
        }
        prev_burst = burst_offset;

        addr_type addr;
        if (allocate_rows.size() > 1) {
            addr = MyAddressAllocator::get_sequence_address(
                burst_offset,
                start_outer_row_loop,
                start_middle_row_loop,
                start_inner_row_loop,
                allocate_rows,
                start_col);
        } else {
            const uint32_t row_base = allocate_rows.empty() ? start_row : allocate_rows[0];
            addr = MyAddressAllocator::get_sequence_address(
                burst_offset,
                start_outer_row_loop,
                start_middle_row_loop,
                start_inner_row_loop,
                row_base,
                start_col);
        }
        addrs.push_back(addr);
    }
    return addrs;
}



uint32_t MyTensor::pim_output_elements_per_bank() {
    // TODO:: Incomplete and Unused
    uint32_t pim_result_elements_per_bank = 0;
    if (_tensor_type == TensorType::WGT) {
        pim_result_elements_per_bank = MyAddressAllocator::bank_allocated_columns_per_tile * allocate_rows.size();
    }
    else if (_tensor_type == TensorType::KCache) {
        pim_result_elements_per_bank = std::ceil((double)_dims[0] / (MyAddressAllocator::KCache_interleaved_banks_per_head * MyAddressAllocator::dram_channels));
    }
    else if (_tensor_type == TensorType::VCache) {
        pim_result_elements_per_bank = std::ceil((double)MyAddressAllocator::d_k / (MyAddressAllocator::VCache_interleaved_banks_per_head * MyAddressAllocator::dram_channels));
    }
    else {
        throw std::runtime_error("unsupported tensor type");
    }
    return pim_result_elements_per_bank;
}


void MyTensor::initial_data_container(DramDataContainer& data_container) {
    // The instance-owned container is sparse: an untouched tensor already
    // reads as zero, so allocating complete DRAM rows here is unnecessary.
    if (!data_container.enabled()) {
        return;
    }
}

void MyTensor::append_data_into_container(const std::vector<uint8_t>& data,
                                          DramDataContainer& data_container) {
    if (data.empty()) {
        return;
    }

    uint64_t tensor_bytes = _precision;
    for (auto dim : _dims) {
        tensor_bytes *= dim;
    }
    if (data.size() > tensor_bytes) {
        throw std::invalid_argument("Tensor data is larger than tensor allocation");
    }

    const uint32_t bytes_per_dram_burst =
        data_container.burst_length() * data_container.dq_bytes();
    assert(bytes_per_dram_burst == MyAddressAllocator::dram_burst_size);

    // Covert row major Byte offset to
    auto row_major_index_from_byte = [this](uint64_t byte_offset) {
        const uint64_t elem_index = byte_offset / _precision;
        std::vector<uint32_t> index(_dims.size(), 0);
        uint64_t remaining = elem_index;
        for (int dim = static_cast<int>(_dims.size()) - 1; dim >= 0; --dim) {
            index[dim] = remaining % _dims[dim];
            remaining /= _dims[dim];
        }
        return index;
    };

    auto sequence_addr_from_byte = [this, bytes_per_dram_burst](uint64_t byte_offset) {
        const uint32_t dram_burst_idx = byte_offset / bytes_per_dram_burst;
        const uint32_t memory_burst_idx = dram_burst_idx / MyAddressAllocator::dram_channels;
        const uint32_t channel_idx = dram_burst_idx % MyAddressAllocator::dram_channels;

        addr_type base_addr;
        if (_tensor_type == TensorType::ACT || _tensor_type == TensorType::PSUM) {
            base_addr = MyAddressAllocator::get_sequence_address(
                memory_burst_idx,
                start_outer_row_loop,
                start_middle_row_loop,
                start_inner_row_loop,
                allocate_rows,
                start_col);
        } else {
            base_addr = MyAddressAllocator::get_sequence_address(
                memory_burst_idx,
                start_outer_row_loop,
                start_middle_row_loop,
                start_inner_row_loop,
                start_row,
                start_col);
        }
        return std::make_pair(MyAddressAllocator::add_channel_index(base_addr, channel_idx),
                              static_cast<uint32_t>(byte_offset % bytes_per_dram_burst));
    };

    auto make_addr = [](uint32_t rank, uint32_t bankgroup, uint32_t bank, uint32_t row, uint32_t col, uint32_t channel) {
        return MyAddressAllocator::make_address_by_index(rank, bankgroup, bank, row, col, channel);
    };

    auto map_weight_ianus = [this, bytes_per_dram_burst, &row_major_index_from_byte, &make_addr](uint64_t byte_offset) {
        if (_dims.size() != 2) {
            throw std::runtime_error("IANUS weight data container mapping supports 2D weight only");
        }
        const auto index = row_major_index_from_byte(byte_offset);
        const uint32_t k = index[0];
        const uint32_t n = index[1];
        const uint32_t elem_byte = byte_offset % _precision;
        const uint64_t column_byte_offset = static_cast<uint64_t>(k) * _precision + elem_byte;
        const uint32_t burst_offset = column_byte_offset / bytes_per_dram_burst;
        const uint32_t byte_in_burst = column_byte_offset % bytes_per_dram_burst;

        uint32_t column_iteration;
        uint32_t bank_offset;
        uint32_t channel_index;
        if (MyAddressAllocator::IANUS_channel_parallel) {
            channel_index = n % MyAddressAllocator::dram_channels;
            const uint32_t channel_column_index = n / MyAddressAllocator::dram_channels;
            column_iteration = channel_column_index / MyAddressAllocator::banks_per_channel;
            bank_offset = channel_column_index % MyAddressAllocator::banks_per_channel;
        } else {
            column_iteration = n / MyAddressAllocator::total_banks;
            const uint32_t total_bank_offset = n % MyAddressAllocator::total_banks;
            channel_index = total_bank_offset / MyAddressAllocator::banks_per_channel;
            bank_offset = total_bank_offset % MyAddressAllocator::banks_per_channel;
        }

        if (column_iteration >= allocate_rows.size()) {
            throw std::runtime_error("IANUS weight mapping exceeds allocated rows");
        }
        const auto& bank = MyAddressAllocator::rababg_bank_index[bank_offset];
        const uint32_t row_index = allocate_rows[column_iteration] + burst_offset / MyAddressAllocator::BL_num_per_row;
        const uint32_t col_index = burst_offset % MyAddressAllocator::BL_num_per_row;
        return std::make_pair(make_addr(bank[0], bank[1], bank[2], row_index, col_index, channel_index), byte_in_burst);
    };

    auto map_weight_dash = [this, bytes_per_dram_burst, &row_major_index_from_byte, &make_addr](uint64_t byte_offset) {
        if (_dims.size() != 2) {
            throw std::runtime_error("DASH weight data container mapping supports 2D weight only");
        }
        const auto index = row_major_index_from_byte(byte_offset);
        const uint32_t k = index[0];
        const uint32_t n = index[1];
        const uint32_t elem_byte = byte_offset % _precision;

        const uint32_t tile_index = n / MyAddressAllocator::tile_width;
        const uint32_t tile_iteration = tile_index / MyAddressAllocator::allocated_tiles_per_iteration;
        const uint32_t tile_offset = tile_index % MyAddressAllocator::allocated_tiles_per_iteration;
        if (tile_iteration >= allocate_rows.size()) {
            throw std::runtime_error("DASH weight mapping exceeds allocated rows");
        }

        const uint32_t n_in_tile = n % MyAddressAllocator::tile_width;
        const uint32_t columns_per_bank = MyAddressAllocator::bank_allocated_columns_per_tile;
        const uint32_t parallel_group = n_in_tile / columns_per_bank;
        const uint32_t channel_index = parallel_group / MyAddressAllocator::interleaved_banks_per_tile;
        const uint32_t interleave_index = parallel_group % MyAddressAllocator::interleaved_banks_per_tile;
        if (channel_index >= MyAddressAllocator::dram_channels) {
            throw std::runtime_error("DASH weight mapping exceeds channel parallelism");
        }

        const uint32_t col_in_bank = n_in_tile % columns_per_bank;
        const uint64_t k_byte_offset = static_cast<uint64_t>(k) * _precision + elem_byte;
        const uint32_t k_burst_offset = k_byte_offset / bytes_per_dram_burst;
        const uint32_t byte_in_burst = k_byte_offset % bytes_per_dram_burst;
        const uint32_t bank_col_offset = k_burst_offset * columns_per_bank + col_in_bank;

        const auto& bank = MyAddressAllocator::interleaved_bank_index[tile_offset][interleave_index];
        const uint32_t row_index = allocate_rows[tile_iteration] + bank_col_offset / MyAddressAllocator::BL_num_per_row;
        const uint32_t col_index = bank_col_offset % MyAddressAllocator::BL_num_per_row;
        return std::make_pair(make_addr(bank[0], bank[2], bank[1], row_index, col_index, channel_index), byte_in_burst);
    };

    auto map_kcache_npu = [this, bytes_per_dram_burst, &row_major_index_from_byte, &make_addr](uint64_t byte_offset) {
        if (_dims.size() != 2) {
            throw std::runtime_error("NPU K/V cache data container mapping supports 2D cache only");
        }
        const auto index = row_major_index_from_byte(byte_offset);
        const uint32_t token = index[0];
        const uint32_t head_index = index[1] / MyAddressAllocator::d_k;
        const uint32_t head_col = index[1] % MyAddressAllocator::d_k;
        const uint32_t elem_byte = byte_offset % _precision;

        const uint32_t entry_index = token / MyAddressAllocator::kv_cache_entry_size;
        const uint32_t token_offset = token % MyAddressAllocator::kv_cache_entry_size;
        const uint32_t total_bank_index = entry_index * MyAddressAllocator::h_kv + head_index;
        const uint32_t row_iteration = total_bank_index / MyAddressAllocator::total_banks;
        const uint32_t row_offset = total_bank_index % MyAddressAllocator::total_banks;
        if (kv_cache_allocate_rows.empty() || row_iteration >= kv_cache_allocate_rows[0].size()) {
            throw std::runtime_error("NPU K/V cache mapping exceeds allocated rows");
        }

        const uint32_t channel_index = row_offset % MyAddressAllocator::dram_channels;
        const uint32_t bank_offset = (row_offset / MyAddressAllocator::dram_channels) % MyAddressAllocator::banks_per_channel;
        const uint64_t head_col_byte = static_cast<uint64_t>(head_col) * _precision + elem_byte;
        const uint32_t head_col_burst = head_col_byte / bytes_per_dram_burst;
        const uint32_t byte_in_burst = head_col_byte % bytes_per_dram_burst;
        const uint32_t col_index = token_offset * MyAddressAllocator::burst_times_per_KCache_row + head_col_burst;

        const auto& bank = MyAddressAllocator::rababg_bank_index[bank_offset];
        return std::make_pair(make_addr(bank[0], bank[1], bank[2], kv_cache_allocate_rows[0][row_iteration], col_index, channel_index), byte_in_burst);
    };

    auto map_kcache_interleaved = [this, bytes_per_dram_burst, &row_major_index_from_byte, &make_addr](uint64_t byte_offset) {
        const auto index = row_major_index_from_byte(byte_offset);
        const uint32_t token = index[0];
        const uint32_t head_index = index[1] / MyAddressAllocator::d_k;
        const uint32_t head_col = index[1] % MyAddressAllocator::d_k;
        const uint32_t elem_byte = byte_offset % _precision;

        const uint32_t channel_index = head_index % MyAddressAllocator::dram_channels;
        const uint32_t channel_head_iteration = (head_index % MyAddressAllocator::allocated_KCache_head_per_iteration) / MyAddressAllocator::dram_channels;
        const uint32_t head_row_index = token / MyAddressAllocator::KCache_rows_per_allocation;
        if (head_index >= kv_cache_allocate_rows.size() || head_row_index >= kv_cache_allocate_rows[head_index].size()) {
            throw std::runtime_error("KCache mapping exceeds allocated rows");
        }

        const uint32_t token_in_allocation = token % MyAddressAllocator::KCache_rows_per_allocation;
        const uint32_t row_offset_in_bank = token_in_allocation / MyAddressAllocator::KCache_interleaved_banks_per_head;
        const uint32_t interleave_index = token_in_allocation % MyAddressAllocator::KCache_interleaved_banks_per_head;
        const auto& bank = MyAddressAllocator::KCache_interleaved_bank_index[channel_head_iteration][interleave_index];

        const uint64_t head_col_byte = static_cast<uint64_t>(head_col) * _precision + elem_byte;
        const uint32_t head_col_burst = head_col_byte / bytes_per_dram_burst;
        const uint32_t byte_in_burst = head_col_byte % bytes_per_dram_burst;
        const uint32_t col_index = row_offset_in_bank * MyAddressAllocator::burst_times_per_KCache_row + head_col_burst;
        return std::make_pair(make_addr(bank[0], bank[2], bank[1], kv_cache_allocate_rows[head_index][head_row_index], col_index, channel_index), byte_in_burst);
    };

    auto map_vcache_interleaved = [this, bytes_per_dram_burst, &row_major_index_from_byte, &make_addr](uint64_t byte_offset) {
        const auto index = row_major_index_from_byte(byte_offset);
        const uint32_t token = index[0];
        const uint32_t head_index = index[1] / MyAddressAllocator::d_k;
        const uint32_t head_col = index[1] % MyAddressAllocator::d_k;
        const uint32_t elem_byte = byte_offset % _precision;

        const uint32_t channel_index = head_index % MyAddressAllocator::dram_channels;
        const uint32_t channel_head_iteration = (head_index % MyAddressAllocator::allocated_VCache_head_per_iteration) / MyAddressAllocator::dram_channels;
        const uint32_t row_unit_iteration = token / MyAddressAllocator::VCache_burst_row_unit;
        const uint32_t head_row_index = row_unit_iteration / MyAddressAllocator::VCache_row_units_per_allocation;
        if (head_index >= kv_cache_allocate_rows.size() || head_row_index >= kv_cache_allocate_rows[head_index].size()) {
            throw std::runtime_error("VCache mapping exceeds allocated rows");
        }

        const uint32_t row_unit_offset = row_unit_iteration % MyAddressAllocator::VCache_row_units_per_allocation;
        const uint32_t interleave_offset = head_col / MyAddressAllocator::VCache_interleaved_banks_per_head;
        const uint32_t interleave_index = head_col % MyAddressAllocator::VCache_interleaved_banks_per_head;
        const uint32_t col_index = row_unit_offset * MyAddressAllocator::VCache_columns_per_bank + interleave_offset;
        const uint32_t byte_in_burst = (token % MyAddressAllocator::VCache_burst_row_unit) * _precision + elem_byte;

        const auto& bank = MyAddressAllocator::VCache_interleaved_bank_index[channel_head_iteration][interleave_index];
        return std::make_pair(make_addr(bank[0], bank[2], bank[1], kv_cache_allocate_rows[head_index][head_row_index], col_index, channel_index), byte_in_burst);
    };

    auto map_byte = [&](uint64_t byte_offset) {
        if (_tensor_type == TensorType::ACT || _tensor_type == TensorType::PSUM || _dims.size() == 1) {
            return sequence_addr_from_byte(byte_offset);
        }

        if (_tensor_type == TensorType::WGT) {
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU ||
                MyAddressAllocator::allocation_scheme == AllocationScheme::NeuPIM) {
                return sequence_addr_from_byte(byte_offset);
            }
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS) {
                return map_weight_ianus(byte_offset);
            }
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                return map_weight_dash(byte_offset);
            }
        }
        else if (_tensor_type == TensorType::KCache) {
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU) {
                return map_kcache_npu(byte_offset);
            }
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS ||
                MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                return map_kcache_interleaved(byte_offset);
            }
        }
        else if (_tensor_type == TensorType::VCache) {
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::NPU) {
                return map_kcache_npu(byte_offset);
            }
            if (MyAddressAllocator::allocation_scheme == AllocationScheme::IANUS ||
                MyAddressAllocator::allocation_scheme == AllocationScheme::DASH) {
                return map_vcache_interleaved(byte_offset);
            }
        }

        throw std::runtime_error("append_data_into_container: unsupported tensor type or allocation scheme");
    };

    std::map<addr_type, std::vector<std::vector<uint8_t>>> pending_writes;
    auto get_or_create_burst = [&pending_writes, &data_container](addr_type addr) -> std::vector<std::vector<uint8_t>>& {
        auto [it, inserted] = pending_writes.emplace(addr, std::vector<std::vector<uint8_t>>{});
        if (inserted) {
            it->second.assign(data_container.burst_length(),
                              std::vector<uint8_t>(data_container.dq_bytes(), 0));
        }
        return it->second;
    };

    for (uint64_t byte_offset = 0; byte_offset < data.size(); ++byte_offset) {
        auto [addr, byte_in_burst] = map_byte(byte_offset);
        if (byte_in_burst >= bytes_per_dram_burst) {
            throw std::runtime_error("append_data_into_container: byte offset exceeds DRAM burst size");
        }
        auto& burst_data = get_or_create_burst(addr);
        const uint32_t col_offset = byte_in_burst / data_container.dq_bytes();
        const uint32_t byte_idx = byte_in_burst % data_container.dq_bytes();
        burst_data[col_offset][byte_idx] = data[byte_offset];
    }

    for (auto& [addr, burst_data] : pending_writes) {
        data_container.write_burst(
            std::move(burst_data),
            MyAddressAllocator::get_channel_index(addr),
            MyAddressAllocator::get_rank_index(addr),
            MyAddressAllocator::get_bankgroup_index(addr),
            MyAddressAllocator::get_bank_index(addr),
            MyAddressAllocator::get_row_index(addr),
            MyAddressAllocator::get_col_index(addr));
    }
}


MyTensor::~MyTensor() {
    // spdlog::info("Remove the current tensor, disable the allocated memory space");
}
