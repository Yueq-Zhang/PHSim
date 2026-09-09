#include "Sram.h"
#include "../common_function.hpp"
#define NUM_PORTS 3

Sram::Sram(const SysConfig& config, const cycle_type &core_cycle, bool accum)
    : _core_cycle(core_cycle) {
    _size = accum ? config.accum_spad_size KB : config.spad_size KB;
    _data_width = config.dram_req_size;
    int precision = config.precision;
    _current_size[0] = 0;
    _current_size[1] = 0;
    _accum = accum;
}

bool Sram::check_hit(addr_type address, int buffer_id) {
    // spdlog::debug("check_hit addr:{:x}, buffer_id:{}, end: {}", address, buffer_id,
    //              _cache_table[buffer_id].find(address) == _cache_table[buffer_id].end());
    if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {  // 如果找的这个地址在cache table中没有, 返回false，无法执行
        return false;
    }

    _cache_table[buffer_id][address].timestamp = _core_cycle; // 如果找到了，设置cache table的time stamp为当前core的运行时间
    return _cache_table[buffer_id][address].valid;
}

bool Sram::check_full(int buffer_id) {
    // return _current_size[buffer_id] < _size / _data_width / 2;
    return _current_size[buffer_id] < _size / 2;
}

bool Sram::check_remain(size_t size, int buffer_id) {   // check remain, 用于检测当前SRAM在分配了一个size之后是否溢出
    // return _current_size[buffer_id] + size <= _size / _data_width / 2;
    return _current_size[buffer_id] + size <= _size / 2;
}

bool Sram::check_allocated(addr_type address, int buffer_id) {   // check allocate, 如果cache table的这个地址已经被分配了，返回就是true，否则返回false
    return _cache_table[buffer_id].find(address) != _cache_table[buffer_id].end();
}

void Sram::cycle() {}

void Sram::flush(int buffer_id) {
    // spdlog::debug("Buffer id {} is flushed.", buffer_id);
    _current_size[buffer_id] = 0;
    _cache_table[buffer_id].clear();
}

void Sram::reserve(addr_type address, int buffer_id, size_t allocated_size, size_t count) {
    // 若被预留的地址不在Cache Table中，更新current size，之后在Cache_Table中创建；如果溢出则报错
    // 若对于Accum SPAD，在Cache Table中可以找到，允许在其基础上进行更新
    if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
        if (!check_remain(allocated_size, buffer_id)) {
            print_all(buffer_id);
            assert(0);
        }
        _current_size[buffer_id] += allocated_size;
    } else if (_cache_table[buffer_id].find(address) != _cache_table[buffer_id].end() && _accum) {  // 当前操作的Acc地址可以在Cache Table中被找到，同时在Accum中, 此时实现size增加扩展
        _cache_table[buffer_id][address].size += allocated_size;
    } else {
        assert(0);
    }
    // spdlog::debug("pushed a entry in cache table with address {} size {} to buffer {}.", address, allocated_size, buffer_id);
    _cache_table[buffer_id][address] = SramEntry{.valid = (count==0), .size = allocated_size, .remain_req_count = count, .timestamp = _core_cycle};  // 创建一个新的
}

void Sram::fill(addr_type address, int buffer_id) {
    assert(check_allocated(address, buffer_id));
    assert(_cache_table[buffer_id][address].remain_req_count > 0);
    assert(!_cache_table[buffer_id][address].valid);
    _cache_table[buffer_id][address].remain_req_count--;
    // spdlog::debug("sram address {:x}, buffer_id: {}, count down to {}", address, buffer_id, _cache_table[buffer_id][address].remain_req_count);
    if (_cache_table[buffer_id][address].remain_req_count == 0) {  // 加载的数据元素向里面填充，如果全部填充了，则对应位置可以是Valid的情况
        _cache_table[buffer_id][address].valid = true;
        // spdlog::debug("Make SRAM Buffer {} Entry address {} Valid", buffer_id, address);
        // spdlog::trace("MAKE valid {} {}F", buffer_id, address);
    }
}

void Sram::count_up(addr_type address, int buffer_id) {
    assert(check_allocated(address, buffer_id));
    _cache_table[buffer_id][address].remain_req_count++;
    // spdlog::debug("sram address {:x} count up to {}", address, _cache_table[buffer_id][address].remain_req_count);
    if (_cache_table[buffer_id][address].valid) {
        _cache_table[buffer_id][address].valid = false;
        spdlog::trace("MAKE valid {} {}", buffer_id, address);
    }
}

void Sram::print_all(int buffer_id) {
    for (auto &[key, val] : _cache_table[buffer_id]) {
        spdlog::debug("{:x} : {}", key, val.size);
    }
}

void Sram::print_non_valid(int buffer_id) {
    for (auto &[key, val] : _cache_table[buffer_id]) {
        if (!val.valid) spdlog::debug("{:x} : {}", key, val.remain_req_count);
    }
}