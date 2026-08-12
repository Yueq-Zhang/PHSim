
#include "MyCore.hpp"

#include <algorithm>
#include <stdexcept>

MyCore::MyCore(uint32_t id, const SysConfig& config):
    _id(id),
    _config(config),
    _core_cycle(0),
    _compute_end_cycle(0),
    _stat_idle_cycle(0),
    _stat_compute_cycle(0),
    _stat_memory_cycle(0),
    _stat_vec_compute_cycle(0),
    _stat_vec_memory_cycle(0),
    _memory_stall_cycle(0),
    _compute_memory_stall_cycle(0),
    _vector_memory_stall_cycle(0),
    _layernorm_stall_cycle(0),
    _rmsnorm_stall_cycle(0),
    _rope_stall_cycle(0),
    _softmax_stall_cycle(0),
    _add_stall_cycle(0),
    _mul_stall_cycle(0),
    _gelu_stall_cycle(0),
    _silu_stall_cycle(0),
    _load_memory_cycle(0),
    _store_memory_cycle(0),
    _stat_vec_idle_cycle(0),
    _stat_gemm_cycle(0),
    _stat_layernorm_cycle(0),
    _stat_rmsnorm_cycle(0),
    _stat_rope_cycle(0),
    _stat_add_cycle(0),
    _stat_mul_cycle(0),
    _stat_gelu_cycle(0),
    _stat_silu_cycle(0),
    _stat_softmax_cycle(0),
    _accum_request_rr_cycle(0),

    _pim_pheader_count(0),
    _pim_gwrite_count(0),
    _pim_comp_count(0),
    _pim_readers_count(0),

    _write_count(0),
    _read_count(0),

    _gemm_count(0),
    _gemv_count(0),
    _layernorm_count(0),
    _rmsnorm_count(0),
    _rope_count(0),
    _softmax_count(0),
    _add_count(0),
    _mul_count(0),
    _gelu_count(0),
    _silu_count(0),
    _im2col_count(0),
    _dummy_count(0),



    _spad(Sram(config, _core_cycle, false)),
    _acc_spad(Sram(config, _core_cycle, true)),
    _pim_spad(Sram(config, _core_cycle, false)),
    _pim_acc_spad(Sram(config, _core_cycle, true)){

    _waiting_write_reqs = 0;
    _waiting_pim_reqs = 0;

    _running_layer = -1;
    _current_spad = 0;
    _current_acc_spad = 0;
    _memory_request_queues.resize(_config.dram_channels);
    _vector_pipelines.resize(_config.vector_core_count);
    auto stat = NPUStat(_core_cycle);
    _stat.push_back(stat);

}

bool MyCore::running() {
    bool running = false;
    running = running || _tiles.size() > 0;
    running = running || !_compute_pipeline.empty();
    running = running || _waiting_write_reqs != 0;
    running = running || _waiting_pim_reqs != 0;
    running = running || !_ld_inst_queue.empty();
    running = running || !_st_inst_queue.empty();
    running = running || !_ex_inst_queue.empty();
    bool temp = running;
    for (auto &vector_pipeline : _vector_pipelines) {
        running = running || !vector_pipeline.empty();
    }
    return running;
}

void MyCore::cycle() {
    // record state every 1000 cycles
    if (_stat.back().start_cycle + 1000 <= _core_cycle) {
        auto stat = NPUStat(_core_cycle);
        _stat.push_back(stat);
    }

    systolic_cycle(); // Matrix Computation
    vector_unit_cycle(); // Vector Computation

    // instruction fetch
    ld_queue_cycle();
    st_queue_cycle();
    ex_queue_cycle();

    pim_queue_cycle();
    // pim instruction fetch
    // pim_ld_queue_cycle();
    // pim_st_queue_cycle();
    // pim_ex_queue_cycle();

    update_stats();

    // update state
    _core_cycle++;
    _spad.cycle();
    _acc_spad.cycle();

    for (auto tile_it = _tiles.begin(); tile_it != _tiles.end();) {
        auto tile = *tile_it;
        if ((tile->remaining_accum_io == 0) && (tile->remaining_computes == 0) && (tile->remaining_loads == 0)
            && (tile->remain_pim_gwrite == 0) && (tile->remain_pim_comp == 0) && (tile->remain_pim_readers == 0)) {
            tile->status = Tile::Status::FINISH;
            _finished_tiles.push(tile);

            tile_it = _tiles.erase(tile_it);
            } else {
                tile_it++;
            }
    }
}

void MyCore::systolic_cycle() {
    /* Compute unit */
    if (!_compute_pipeline.empty() && _compute_pipeline.front().finish_cycle <= _core_cycle) { 
        // ִexecute the first instruction of computation_Pipeline
        Instruction &inst = _compute_pipeline.front();
        // update result stratch pad
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            // spdlog::info("Matrix instruction finished, instruction info {}, spad_id:{}", inst.repr(), inst.inst_information, inst.accum_spad_id);
            _acc_spad.fill(inst.dest_addr, inst.accum_spad_id); // update accum_spad after instruction finished
        } else {
            assert(0);
        }
        // update the tile state
        if (auto tile = inst.parent_tile.lock()) {
            assert(std::find(_tiles.begin(), _tiles.end(), tile) != _tiles.end());
            tile->remaining_accum_io--;
            tile->remaining_computes--;
        } else {
            assert(0);
        }
        _compute_pipeline.pop();
    }
}

void MyCore::vector_unit_cycle() {
    for (auto &vector_pipeline : _vector_pipelines) {  // multi VU is enabled
        if (!vector_pipeline.empty() && vector_pipeline.front().finish_cycle <= _core_cycle) {
            Instruction &inst = vector_pipeline.front();
            Sram *buffer = inst.is_pim_inst ? &_pim_acc_spad : &_acc_spad;
            if (inst.dest_addr >= ACCUM_SPAD_BASE) {
                buffer->fill(inst.dest_addr, inst.accum_spad_id);
            } else {
                assert(0);
            }
            if (auto tile = inst.parent_tile.lock()) {
                assert(std::find(_tiles.begin(), _tiles.end(), tile) != _tiles.end());  // current tile is an operation tile of current core
                tile->remaining_accum_io--;
                tile->remaining_computes--;
            } else {
                assert(0);
            }
            vector_pipeline.pop();
        }
    }
}

void MyCore::ld_queue_cycle() {
    /* LD instruction queue */
    // todo: ld_queue.cycle();
    std::vector<uint32_t> ch_req_dist(_config.dram_channels, 0);
    bool filled = false;
    while (!_ld_inst_queue.empty()) {
        Instruction &front = _ld_inst_queue.front();
        if (front.opcode == Opcode::MOVIN) {
            // bool prefetched = false;
            Sram *buffer;
            int buffer_id;
            if (front.dest_addr >= ACCUM_SPAD_BASE) {
                buffer = &_acc_spad;
                buffer_id = front.accum_spad_id;
            } else {
                buffer = &_spad;
                buffer_id = front.spad_id;
            }

            if (front.skip==true or front.src_addrs.empty()) {
                spdlog::info("Current load instruction has no src address, just reserve the dest_addr {}", front.dest_addr);
                buffer->reserve(front.dest_addr, buffer_id, front.size, 0); // remain burst times of cache block of SRAM
                _ld_inst_queue.pop();
                front.parent_tile.lock()->remaining_loads--;
                continue;
            }

            assert(!front.src_addrs.empty()); // Generate Read memory instruction for data loading
            // TODO: improve the trace scale to Cache Line
            auto my_accesses = MemoryAccess::gen_trace_from_instruction(
                front, generate_mem_access_id(), MyAddressAllocator::dram_burst_size, MemoryAccessType::READ,
                true, _id, _core_cycle, buffer_id, StagePlatform::SA);

            // Reserve the place in SPAD, and update the state of Tile
            buffer->reserve(front.dest_addr, buffer_id, front.size, my_accesses.size());
            if (auto tile = front.parent_tile.lock()) {
                tile->remaining_loads += my_accesses.size() - 1;
                tile->stat.memory_reads += my_accesses.size() * MyAddressAllocator::dram_burst_size;
            } else {
                assert(0);
            }
            // Append the Read memory requests to memory request queue
            for (auto& access : my_accesses) {
                filled = true;
                ch_req_dist[MyAddressAllocator::get_channel_index(access->dram_address)]++;
                push_memory_request(std::move(access));
            }
            _ld_inst_queue.pop();
            _read_count += my_accesses.size();
        }
        else {
            assert(0);
        }
    }
}


void MyCore::st_queue_cycle() {
    /* ST instruction queue */
    // todo: st_queue.cycle();
    if (!_st_inst_queue.empty()) {
        Instruction &front = _st_inst_queue.front();
        Sram *buffer;
        int buffer_id;
        if (front.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_acc_spad;
            buffer_id = front.accum_spad_id;
        } else {
            buffer = &_spad;
            buffer_id = front.spad_id;
        }

        if (front.skip==true or front.src_addrs.empty()) {
            spdlog::info("Current store instruction has no Destination address");
            _st_inst_queue.pop();
            front.parent_tile.lock()->remaining_accum_io--;
            return;
        }

        ast(!front.src_addrs.empty());
        // MOVOUT can be executed once all data are ready in buffer
        if (buffer->check_hit(front.dest_addr, buffer_id) && (front.opcode == Opcode::MOVOUT || front.opcode == Opcode::MOVOUT_POOL) && _waiting_pim_reqs == 0) {
            auto my_accesses = MemoryAccess::gen_trace_from_instruction(
                front, generate_mem_access_id(), MyAddressAllocator::dram_burst_size, MemoryAccessType::WRITE,
                true, _id, _core_cycle, buffer_id, StagePlatform::SA);

            if (auto tile = front.parent_tile.lock()) {
                tile->remaining_accum_io += my_accesses.size() - 1;
                tile->stat.memory_writes += my_accesses.size() * MyAddressAllocator::dram_burst_size;
            } else {
                assert(0);
            }
            for (auto& access : my_accesses) {
                push_memory_request(std::move(access));
                _waiting_write_reqs++;
            }
            _st_inst_queue.pop();
            _write_count += my_accesses.size();
        }
    }
}

void MyCore::ex_queue_cycle() {
    /* EX instruction queue */
    if (!_ex_inst_queue.empty()) {
        Instruction ready_inst = get_first_ready_ex_inst();
        if (ready_inst.valid)
            issue_ex_inst(ready_inst);
        else {
            /* Update memory stall stat */
        }
    }
}

// pipelines
Instruction MyCore::get_first_ready_ex_inst() {
    Instruction inst = _ex_inst_queue.front();
    if (can_issue_compute(inst)) {
        // spdlog::info("The executed instruction {} is executed with src address {} and dest address {}", inst.print_optype(), inst.src_addrs, inst.dest_addr);
        _ex_inst_queue.pop();
        return std::move(inst);
    }

    return Instruction{.valid = false};
}


void MyCore::pim_queue_cycle() {
    /* PIM instruction queue */
    // PIM Instruction Count
    while (!_pim_inst_queue.empty()) {
        Instruction &front = _pim_inst_queue.front();
        if (front.opcode == Opcode::PIM_HEADER) {
            if (auto tile = front.parent_tile.lock()) {
                if (tile->remaining_loads == 0) {
                    // Update PIM Start Cycle
                    if (tile->pim_start_cycle == 0) {
                        tile->pim_start_cycle = _core_cycle;
                    }
                    // spdlog::info("All MOVEIN Instruction is finished, Issued PIM instruction is PIM_Header");
                    auto my_accesses = MemoryAccess::gen_pim_trace_from_instruction(front, generate_mem_access_id(), 0,
                        MemoryAccessType::P_HEADER,true, _id, _core_cycle, tile->spad_id, StagePlatform::PIM);
                    for (auto& access : my_accesses) {
                        push_memory_request(std::move(access));
                        _waiting_pim_reqs++;
                    }
                    // tile->pim_inst_count += my_accesses.size();  Pheader is not take into account
                    _pim_inst_queue.pop();
                    _pim_pheader_count += my_accesses.size();
                }
                else {
                    // spdlog::info("After {} Loads, PIM_Header in current tile can issued", tile->remaining_loads);
                    break;
                }
            }
            else {
                assert(0);
            }
        }
        else if (front.opcode == Opcode::PIM_GWRITE) {
            //spdlog::info("After execute the PIM_Header, gwrite Instruction can executed");
            if (auto tile = front.parent_tile.lock()) {
                auto my_accesses = MemoryAccess::gen_pim_trace_from_instruction(front, generate_mem_access_id(), 0,
                    MemoryAccessType::GWRITE,true, _id, _core_cycle, tile->spad_id, StagePlatform::PIM);

                for (auto& access : my_accesses) {
                    push_memory_request(std::move(access));
                    _waiting_pim_reqs++;
                }
                // tile->pim_inst_count += front.src_addrs.size();
                tile->remain_pim_gwrite += my_accesses.size() - 1;
                _pim_inst_queue.pop();
                _pim_gwrite_count += my_accesses.size();
            }
            else {
                assert(0);
            }
        }
        else if (front.opcode == Opcode::PIM_COMP) {
            if (auto tile = front.parent_tile.lock()) {
                if (tile->remain_pim_gwrite!=0) {
                    // spdlog::info("The PIM_GWRITE of current tile is not finished, {} Exist", tile->remain_pim_gwrite);
                    break;
                }
                // spdlog::info("After all the PIM input data was written by PIM_GWRITE, Begin PIM_COMP");
                auto my_accesses = MemoryAccess::gen_pim_trace_from_instruction(front, generate_mem_access_id(), 0,
                    MemoryAccessType::COMP,true, _id, _core_cycle, tile->spad_id, StagePlatform::PIM);
                for (auto& access : my_accesses) {
                    push_memory_request(std::move(access));
                    _waiting_pim_reqs++;
                }
                if (PIM_Parameters::dual_bank) {
                    tile->pim_inst_count += front.src_addrs.size() * 2;
                }
                else {
                    tile->pim_inst_count += front.src_addrs.size();
                }
                  // Channel Separate for Comp Instruction
                tile->remain_pim_comp += my_accesses.size() - 1;
                _pim_inst_queue.pop();
                _pim_comp_count += my_accesses.size();
            }
            else {
                assert(0);
            }
        }
        else if (front.opcode == Opcode::PIM_COMP_HASH) {
            if (auto tile = front.parent_tile.lock()) {
                if (tile->remain_pim_gwrite!=0) {
                    // spdlog::info("The PIM_GWRITE of current tile is not finished, {} Exist", tile->remain_pim_gwrite);
                    break;
                }
                // spdlog::info("After all the PIM input data was written by PIM_GWRITE, Begin PIM_COMP");
                auto my_accesses = MemoryAccess::gen_pim_trace_from_instruction(front, generate_mem_access_id(), 0,
                    MemoryAccessType::COMP_HASH,true, _id, _core_cycle, tile->spad_id, StagePlatform::PIM);
                for (auto& access : my_accesses) {
                    push_memory_request(std::move(access));
                    _waiting_pim_reqs++;
                }
                tile->pim_inst_count += (my_accesses.size()/ MyAddressAllocator::total_banks) * (MyAddressAllocator::AddrGranularity_Hash_Bytes / MyAddressAllocator::dram_burst_size);
                /*
                if (PIM_Parameters::dual_bank) {
                    tile->pim_inst_count += front.src_addrs.size() * 2;
                }
                else {
                    tile->pim_inst_count += front.src_addrs.size();
                }
                */
                // Channel Separate for Comp Instruction
                tile->remain_pim_comp += my_accesses.size() - 1;
                _pim_inst_queue.pop();
                _pim_comp_count += my_accesses.size();
            }
            else {
                assert(0);
            }
        }
        else if (front.opcode == Opcode::PIM_READRES) {
            if (auto tile = front.parent_tile.lock()) {
                if (tile->remain_pim_comp!=0) {
                    // spdlog::info("The PIM_COMP of current tile is not finished, {} Exist", tile->remain_pim_comp);
                    break;
                }
                // spdlog::info("After all the PIM COMP was Finished, Begin PIM_READERS");
                // bool prefetched = false;
                Sram *buffer;
                int buffer_id;
                if (front.dest_addr >= ACCUM_SPAD_BASE) {
                    buffer = &_acc_spad;
                    buffer_id = front.accum_spad_id;
                } else {
                    buffer = &_spad;
                    buffer_id = front.spad_id;
                }

                auto my_accesses = MemoryAccess::gen_pim_trace_from_instruction(front, generate_mem_access_id(), 0,
                    MemoryAccessType::READRES,true, _id, _core_cycle, buffer_id, StagePlatform::PIM);

                for (auto& access : my_accesses) {
                    push_memory_request(std::move(access));
                    _waiting_pim_reqs++;
                }

                // tile->pim_inst_count += front.src_addrs.size();
                // buffer->reserve(front.dest_addr, buffer_id, front.size, my_accesses.size());
                if (buffer->check_allocated(front.dest_addr, buffer_id)) {
                    for (auto i=0 ; i<my_accesses.size() ; i++) {
                        buffer->count_up(front.dest_addr, front.accum_spad_id);
                        // spdlog::info("PIM Readers the READERS dest_addr of buffer is allocated on chip PSUM Accumulation, count up");
                    }
                }
                else{
                    buffer->reserve(front.dest_addr, buffer_id, front.size, my_accesses.size());
                }

                tile->remain_pim_readers += my_accesses.size() - 1;
                _pim_inst_queue.pop();
                _pim_readers_count += my_accesses.size();
            }
            else {
                assert(0);
            }
        }
        else {
            throw std::runtime_error("Current Instruction is no support");
        }
    }

}


// Check if the tile can be issued to Core for Computation
bool MyCore::can_issue(Tile &next_tile) {
    if (next_tile.pim_tile) {
        if (_id != 0) {
            return false;
        }
    }

    if (_tiles.empty()) {
        return true;
    }

    auto next_spad = _current_spad ^ 1;
    auto next_acc_spad = _current_acc_spad ^ 1;
    // load and compute should be over at the other side, PingPong Buffer, two tile can be executed, but next Buffer should valid
    for (auto tile : _tiles) {
        if (tile->spad_id == next_spad) {
            if ((tile->remaining_loads != 0) || (tile->remaining_computes != 0) || (tile->remain_pim_comp != 0) || (tile->remaining_accum_io != 0))  {
                // spdlog::info("issue failed. accum true. spad id {}, remaining load = {}, remaining_computes = {}, remaining pim comp = {}",
                // tile->spad_id, tile->remaining_loads, tile->remaining_computes, tile->remain_pim_comp);
                return false;
            }
        }
    }

    if (!next_tile.accum) {
        // load, compute and store should be over at the other side
        for (auto tile : _tiles) {
            if (tile->accum_spad_id == next_acc_spad) {
                if (tile->remaining_accum_io != 0) {
                    // spdlog::info("issue failed. accum false. acc spad id {}, remaining_accum_io number is {}", tile->accum_spad_id, tile->remaining_accum_io);
                    return false;
                }
            }
        }
    }
    // spdlog::info("issue succeeded. accum {}, spad id {}, acc spad id {}", next_tile.accum ? "true" : "false", next_spad, next_acc_spad);
    if (_tiles.size() < 2) {
        return true;
    }
    else {
        return false;
    }
}

// todo: check tile start cycle
void MyCore::issue(Tile &in_tile) {
    auto tile = std::make_shared<Tile>(in_tile);
    if (tile->pim_tile){assert(_id == 0);}
    tile->stat = TileStat(_core_cycle);  // Initial current tile
    if (tile->skip) {
        tile->status = Tile::Status::FINISH;
        _finished_tiles.push(tile);
        return;
    }
    /* Double buffer */
    _current_spad = (_current_spad + 1) % 2;
    _spad.flush(_current_spad);
    tile->spad_id = _current_spad;
    if (!tile->accum) {
        /* Accumeulate tile uses same acc spad buffer */
        // accumulate to same acc_spad if K > 1
        _current_acc_spad = (_current_acc_spad + 1) % 2;
        _acc_spad.flush(_current_acc_spad);
    }
    tile->accum_spad_id = _current_acc_spad;
    tile->status = Tile::Status::RUNNING;
    if (_running_layer != tile->operation_id) {
        _running_layer = tile->operation_id;
    }

    // Put Load, Compute and Storeָ instructions into _ld_inst_queue, _ex_inst_queue, _st_inst_queue
    // Put PIM instruction into _pim_inst_queue   
    tile->remaining_loads = 0;
    tile->remaining_computes = 0;
    tile->remaining_accum_io = 0;
    tile->remain_pim_gwrite = 0;
    tile->remain_pim_comp = 0;
    tile->remain_pim_readers = 0;

    uint32_t load_trace = 0;
    uint32_t store_trace = 0;

    for (auto &inst : tile->instructions) {
        inst.parent_tile = std::weak_ptr<Tile>(tile);
        inst.spad_id = tile->spad_id;
        inst.accum_spad_id = tile->accum_spad_id;
        Sram *buffer;
        int buffer_id;
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_acc_spad;
            buffer_id = tile->accum_spad_id;
        } else {
            buffer = &_spad;
            buffer_id = tile->spad_id;
        }

        if (inst.opcode == Opcode::MOVIN) {  // Load instruction
            if (inst.src_addrs.empty()) {
                _ld_inst_queue.push(inst);  // MOVIN with no source addr
            }
            else if (!buffer->check_allocated(inst.dest_addr, buffer_id) && buffer->check_remain(inst.size, buffer_id)) {
                // check the SRAM not overflow
                tile->remaining_loads++;
                _ld_inst_queue.push(inst);
                // The total generated trace number
                if (inst.per_ch_inst) {
                    load_trace += inst.src_addrs.size();
                }
                else {
                    load_trace += inst.src_addrs.size();
                }
            }
            else {
                // SRAM overflow
                spdlog::info("sram size: {} / sram used: {}", _config.spad_size KB / 2,  buffer->get_current_size(buffer_id));
                spdlog::info("instruction destination address {:x}", inst.dest_addr);
                spdlog::info("failed to allocate {} on sram.", inst.size);
                buffer->print_all(buffer_id);
                /*Invalid state */
                assert(0);
            }
        }
        else if (inst.opcode == Opcode::MOVOUT || inst.opcode == Opcode::MOVOUT_POOL) {  // Store Instructions
            if (!inst.src_addrs.empty()) {
                tile->remaining_accum_io++;
                if (inst.per_ch_inst) {
                    store_trace += inst.src_addrs.size();
                }
                else {
                    store_trace += inst.src_addrs.size() * MyAddressAllocator::dram_channels;
                }
            }
            _st_inst_queue.push(inst);
        }
        else if (inst.opcode == Opcode::PIM_HEADER || inst.opcode == Opcode::PIM_GWRITE || inst.opcode == Opcode::PIM_COMP  ||
            inst.opcode == Opcode::PIM_COMP_HASH || inst.opcode == Opcode::PIM_READRES) {
            if (inst.opcode == Opcode::PIM_GWRITE) {
                tile->remain_pim_gwrite++;
            }
            else if (inst.opcode == Opcode::PIM_COMP || inst.opcode == Opcode::PIM_COMP_HASH) {
                tile->remain_pim_comp++;
            }
            else if (inst.opcode == Opcode::PIM_READRES) {
                tile->remain_pim_readers++;
            }
            _pim_inst_queue.push(inst);
        }
        else {
            /* Ex inst queue */
            tile->remaining_accum_io++;
            tile->remaining_computes++;
            _ex_inst_queue.push(inst);
        }
    }
    spdlog::info("Tile of Operation: {} issued for Core {}, Spad {} and Acc_Spad {}, contains {} Load, {} Execution and {} Store, {} Load trace and {} Write Trace",
        in_tile.optype, _id, _current_spad ,_current_acc_spad,
        tile->remaining_loads, tile->remaining_computes, tile->remaining_accum_io - tile->remaining_computes, load_trace, store_trace);
    if (tile->pim_tile) {
        spdlog::info("Current Tile is a pim tile, contains {} PIM_GWRITE, {} PIM_COMP, {} PIM_READERS Trace",
            tile->remain_pim_gwrite, tile->remain_pim_comp, tile->remain_pim_readers);
    }

    _tiles.push_back(tile);
}

bool MyCore::can_issue_pim() {return _pim_tiles.empty();}

void MyCore::issue_pim(Tile &in_tile) {
    spdlog::info("pim tile issued {}", in_tile.repr());
    auto tile = std::make_shared<Tile>(in_tile);
    tile->stat = TileStat(_core_cycle);
    if (tile->skip) {
        tile->status = Tile::Status::FINISH;
        _finished_tiles.push(tile);
        return;
    }

    _pim_spad.flush(0);
    _pim_acc_spad.flush(0);

    tile->spad_id = 0;
    tile->accum_spad_id = 0;
    tile->status = Tile::Status::RUNNING;
    if (_running_layer != tile->operation_id) {
        _running_layer = tile->operation_id;
    }

    tile->remaining_loads = 0;
    tile->remaining_computes = 0;
    tile->remaining_accum_io = 0;
    for (auto &inst : tile->instructions) {
        inst.is_pim_inst = true;
        inst.parent_tile = std::weak_ptr<Tile>(tile);
        inst.spad_id = tile->spad_id;
        inst.accum_spad_id = tile->accum_spad_id;
        Sram *buffer;
        int buffer_id;
        if (inst.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_pim_acc_spad;
            buffer_id = tile->accum_spad_id;
        } else {
            buffer = &_pim_spad;
            buffer_id = tile->spad_id;
        }
        if (inst.opcode == Opcode::PIM_HEADER) {
            _ld_inst_queue_for_pim.push(inst);
        } else if (inst.opcode == Opcode::MOVIN || inst.opcode == Opcode::PIM_GWRITE ||
                   inst.opcode == Opcode::PIM_COMP || inst.opcode == Opcode::PIM_COMPS_READRES) {
            if (!buffer->check_allocated(inst.dest_addr, buffer_id) &&  buffer->check_remain(inst.size, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue_for_pim.push(inst);
            } else {
                spdlog::info("check_allocated: {}", buffer->check_allocated(inst.dest_addr, buffer_id));
                spdlog::info("check_remain: {}", buffer->check_remain(inst.size, buffer_id));
                spdlog::info("sram size: {} / sram used: {}", _config.spad_size KB / 2, buffer->get_current_size(buffer_id));
                spdlog::info("instruction destination address {:x}", inst.dest_addr);
                spdlog::info("failed to allocate {} on sram.", inst.size);
                buffer->print_all(buffer_id);
                /*Invalid state */
                assert(0);
            }
        } else if (inst.opcode == Opcode::PIM_READRES) {
            if (buffer->check_allocated(inst.dest_addr, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue_for_pim.push(inst);
            }
            else if (!buffer->check_allocated(inst.dest_addr, buffer_id) && buffer->check_remain(inst.size, buffer_id)) {
                tile->remaining_loads++;
                _ld_inst_queue_for_pim.push(inst);
            }
        } else if (inst.opcode == Opcode::MOVOUT || inst.opcode == Opcode::MOVOUT_POOL) {
            tile->remaining_accum_io++;
            _st_inst_queue_for_pim.push(inst);
        } else {
            /* Ex inst queue */
            tile->remaining_accum_io++;
            tile->remaining_computes++;
            _ex_inst_queue_for_pim.push(inst);
        }
    }
    // spdlog::info("tile pushed to core._tiles {}", tile.repr());
    _pim_tiles.push_back(tile);
}

// return EMPTY or FINISHED
Ptr<Tile> MyCore::pop_finished_tile() {
    if (_finished_tiles.empty()) {
        return nullptr;
    }

    auto result = _finished_tiles.front();
    result->stat.end_cycle = _core_cycle;
    _finished_tiles.pop();
    return result;
}


void MyCore::push_memory_request(std::unique_ptr<MemoryAccess> request) {
    push_memory_request(_memory_access_owner.adopt(std::move(request)));
}

void MyCore::push_memory_request(MemoryAccess *request) {
    //if (request->req_type == MemoryAccessType::P_HEADER || request->req_type == MemoryAccessType::GWRITE || request->req_type == MemoryAccessType::COMP || request->req_type == MemoryAccessType::READRES) {
    uint32_t channel_index = MyAddressAllocator::get_channel_index(request->dram_address);
    _memory_request_queues[channel_index].push(request);
}

void MyCore::push_memory_response(MemoryAccess *response) {
    // Update SRAM based on the memory response
    // Updatae the state of the parent tile and spad, if store, remaining_accum_io--, if load, remaining_loads--
    assert(!response->request);  // can only push response
    Sram *acc_spad = &_acc_spad;
    Sram *spad = &_spad;
    uint32_t buf_id;

    bool is_write = response->req_type == MemoryAccessType::WRITE;
    bool is_read = response->req_type == MemoryAccessType::READ;
    if (auto tile = response->parent_tile.lock()) {
        assert(std::find(_tiles.begin(), _tiles.end(), tile) != _tiles.end());
        if (is_write) {
            tile->remaining_accum_io--;
        }
        else if (response->req_type == MemoryAccessType::GWRITE) {
            tile->remain_pim_gwrite--;
        }
        else if (response->req_type == MemoryAccessType::COMP || response->req_type == MemoryAccessType::COMP_HASH) {
            tile->remain_pim_comp--;
        }
        else if (response->req_type == MemoryAccessType::READRES) {
            tile->remain_pim_readers--;
        }
        else if (is_read) {
            tile->remaining_loads--;
        }
        /*
        if (tile->pim_finish_cycle == 0 && tile->pim_inst_count > 0 && tile->remain_pim_gwrite == 0 && tile->remain_pim_comp == 0 && tile->remain_pim_readers == 0) {
            tile->pim_finish_cycle = _core_cycle;
        }
        */
        if (tile->pim_finish_cycle == 0 && tile->pim_inst_count > 0 && tile->remain_pim_comp == 0) {
            tile->pim_finish_cycle = _core_cycle;
        }
    } else {
        assert(0);
    }

    if (is_write) {
        _waiting_write_reqs--;
    }
    else if (response->req_type == MemoryAccessType::P_HEADER || response->req_type == MemoryAccessType::GWRITE ||
        response->req_type == MemoryAccessType::COMP || response->req_type == MemoryAccessType::COMP_HASH) {
        // pim_header, pim_gwrite, pim_comp
        // spdlog::info("Receive response of PIM Instruction : {}", memAccessTypeString(response->req_type));
        _waiting_pim_reqs--;
    }
    else if (response->req_type == MemoryAccessType::READRES) {
        // spdlog::info("Receive response of PIM Instruction : {}", memAccessTypeString(response->req_type));
        _waiting_pim_reqs--;
        if (response->spad_address >= ACCUM_SPAD_BASE) {
            acc_spad->fill(response->spad_address, response->buffer_id);
        }
        else {
            spad->fill(response->spad_address, response->buffer_id);
        }
    }
    else if (response->spad_address >= ACCUM_SPAD_BASE) {
        // spdlog::info("{} response to accum_spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit
        // case2: load bias to _accum_spad
        acc_spad->fill(response->spad_address, response->buffer_id);
    }
    else {
        // spdlog::info("{} response to _spad, cycle:{}", is_read ? "LOAD" : "GEMV",
        //              _core_cycle);  // >>> gsheo: remove it before commit
        // case3: load activation or weight to _spad
        spad->fill(response->spad_address, response->buffer_id);
    }
    _memory_access_owner.release(response);
}


// checks if inputs are loaded.
bool MyCore::can_issue_compute(Instruction &inst) {
    bool result = true;

    // src addr: spad key
    for (addr_type addr : inst.src_addrs) {
        if (inst.src_from_accum && addr >= ACCUM_SPAD_BASE) {
            result = result && _acc_spad.check_hit(addr, inst.accum_spad_id);
            continue;
        }

        result = result && _spad.check_hit(addr, inst.spad_id);
    }
    if (!result) {
        for (addr_type addr : inst.src_addrs) {
            // spdlog::info("Core[{}] Dependency fail : {:x} , {} for {}", _id, addr,
            //              _spad.check_hit(addr, inst.spad_id), inst.repr());
        }
    }
    // spdlog::info("can_issue_compute: {} {}", result ? "okay" : "nope", inst.repr());
    return result;
}

bool MyCore::pim_can_issue_compute(Instruction &inst) {
    bool result = true;

    // src addr: spad key
    for (addr_type addr : inst.src_addrs) {
        if (inst.src_from_accum && addr >= ACCUM_SPAD_BASE) {
            result = result && _pim_acc_spad.check_hit(addr, inst.accum_spad_id);
            continue;
        }
        result = result && _pim_spad.check_hit(addr, inst.spad_id);
    }
    if (!result) {
        for (addr_type addr : inst.src_addrs) {
            // spdlog::info("NeuPIMSCore[{}] Dependency fail : {:x} , {} for {}", _id, addr,
            //              _spad.check_hit(addr, inst.spad_id), inst.repr());
        }
    }
    // spdlog::info("can_issue_compute: {} {}", result ? "okay" : "nope", inst.repr());
    return result;
}

cycle_type MyCore::get_inst_compute_cycles(Instruction &inst) {
    return _config.core_height + _config.core_width - 2 + MAX(inst.size, 4);
}

cycle_type MyCore::get_vector_compute_cycles(Instruction &inst) {
    cycle_type vec_op_iter = calculate_vector_op_iterations(inst.size);
    cycle_type add_tree_iter = calculate_add_tree_iterations(inst.size);
    cycle_type add_tree, scalar_ops, vector_ops;
    switch (inst.opcode) {
        case Opcode::LAYERNORM:
            add_tree = 2 * add_tree_iter * _config.add_tree_latency;
            scalar_ops = 2 * _config.scalar_mul_latency + _config.scalar_sqrt_latency;
            // 1 addition, 1 subtraction, 1 division, 2 multiplication.
            vector_ops = vec_op_iter * (2 * _config.add_latency + 3 * _config.mul_latency);
            return add_tree + scalar_ops + vector_ops;
        case Opcode::RMSNORM:
            add_tree = add_tree_iter * _config.add_tree_latency;
            scalar_ops = _config.scalar_mul_latency + _config.scalar_sqrt_latency;
            vector_ops = vec_op_iter * (3 * _config.mul_latency);
            return add_tree + scalar_ops + vector_ops;
        case Opcode::ROPE:
            return vec_op_iter * (2 * _config.mul_latency + _config.add_latency);
        case Opcode::SOFTMAX:
            // 1 add tree, 1 compare tree
            add_tree = 2 * add_tree_iter * _config.add_tree_latency;
            vector_ops = vec_op_iter * (_config.add_latency + _config.exp_latency + _config.mul_latency);
            return add_tree + vector_ops;
        case Opcode::ADD:
            return vec_op_iter * _config.add_latency;
        case Opcode::MUL:
            return vec_op_iter * _config.mul_latency;
        case Opcode::GELU:
            return vec_op_iter * _config.gelu_latency;
        case Opcode::SILU:
            return vec_op_iter * (_config.exp_latency + _config.add_latency + _config.mul_latency);
        case Opcode::GEMV:
            add_tree = add_tree_iter * _config.add_tree_latency;
            vector_ops = vec_op_iter * _config.mul_latency;
            return add_tree + vector_ops;
        case Opcode::DATA_CONVERT:
        case Opcode::DUMMY:
            return 1;
        default:
            spdlog::error("not configured operation. {}", inst.id);
            throw std::invalid_argument("Unsupported opcode in MyCore vector cycle calculation");
    }
}


cycle_type MyCore::calculate_vector_op_iterations(uint32_t vector_size) {
    uint32_t calculation_unit = _config.vector_core_width;
    uint32_t ret = vector_size / calculation_unit;
    if (vector_size % calculation_unit != 0) {
        ret++;
    }
    return ret;
}


cycle_type MyCore::calculate_add_tree_iterations(uint32_t vector_size) {
    uint32_t calculation_unit = _config.vector_core_width;
    if (vector_size <= calculation_unit) {
        return 1;
    }

    uint32_t ret = vector_size / calculation_unit;
    if (vector_size % calculation_unit != 0) {
        ret++;
    }
    return ret + calculate_add_tree_iterations(ret);
}


void MyCore::issue_ex_inst(Instruction inst) {
    // spdlog::info("cycle:{}, {}", _core_cycle, inst.repr());
    if (inst.opcode == Opcode::GEMM || inst.opcode == Opcode::GEMM_PRELOAD) { // Computation for Matrix Unit
        auto parent_tile = inst.parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        // tile_m/tile_k/tile_n are tile indices in the current generator, not
        // dimensions.  Count padded MAC work performed by the physical array.
        parent_tile->stat.num_calculation +=
            static_cast<uint64_t>(_config.core_height) * _config.core_width *
            inst.size;
        parent_tile->stat.weight_load_cycles += _config.core_height * _config.core_width * _config.vector_core_width;

        if (inst.opcode == Opcode::GEMM_PRELOAD) {
            _stat_systolic_preload_issue_count++;
        }
        if (!_compute_pipeline.empty()) {
            /* Preload can be hided */
            uint32_t offset = _compute_pipeline.back().size;
            // xxx why 4?
            // maybe pushing to the systolic array input queue. 4 cycles to start?
            offset = MAX(offset, 4);
            if (inst.opcode == Opcode::GEMM_PRELOAD) {
                // State mul-pre
                parent_tile->stat.weight_load_cycles += _config.core_height;
                offset = _config.core_height;
            }
            inst.start_cycle = _compute_pipeline.back().start_cycle + offset;
        } else {
            inst.start_cycle = _core_cycle;
            /* Preload weight to systolic array*/
            if (inst.opcode == Opcode::GEMM_PRELOAD) {
                /* Weight preload from buffer latecny + Weight preload latency */
                inst.start_cycle += _config.core_height + _config.core_height - 1;
            }
        }
        auto MU_compute_cycle = get_inst_compute_cycles(inst);
        inst.finish_cycle = inst.start_cycle + MU_compute_cycle;
        // spdlog::info("finish_cycle: {}", inst.finish_cycle);
        _compute_pipeline.push(inst);
        _stat_systolic_inst_issue_count++;
    }
    else if (inst.opcode == Opcode::IM2COL || inst.opcode == Opcode::LAYERNORM || inst.opcode == Opcode::RMSNORM || inst.opcode == Opcode::ROPE || inst.opcode == Opcode::SOFTMAX ||
             inst.opcode == Opcode::GEMV || inst.opcode == Opcode::ADD || inst.opcode == Opcode::MUL || inst.opcode == Opcode::GELU || inst.opcode == Opcode::SILU ||
             inst.opcode == Opcode::DATA_CONVERT || inst.opcode == Opcode::DUMMY) {  // Computation for Vector Unit
        if (inst.opcode != Opcode::IM2COL && inst.opcode != Opcode::DATA_CONVERT &&
            inst.opcode != Opcode::DUMMY) {
            auto parent_tile = inst.parent_tile.lock();
            assert(parent_tile != nullptr);
            parent_tile->stat.num_calculation +=
                calculate_vector_op_iterations(inst.size) * _config.vector_core_width;
        }
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        std::queue<Instruction> *least_filled_vpu;
        cycle_type finish_cycle = std::numeric_limits<uint64_t>::max();
        for (auto &vector_pipeline : _vector_pipelines) {
            if (vector_pipeline.empty()) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
                break;
            }
            if (vector_pipeline.back().finish_cycle < finish_cycle) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
            }
        }
        inst.start_cycle = finish_cycle;
        auto VU_compute_cycle = get_vector_compute_cycles(inst);
        inst.finish_cycle = inst.start_cycle + VU_compute_cycle;
        least_filled_vpu->push(inst);
    }

    // Store the computation result
    if (_acc_spad.check_allocated(inst.dest_addr, inst.accum_spad_id)) { // if dest_addr is on sram, count up.
        _acc_spad.count_up(inst.dest_addr, inst.accum_spad_id);        // spdlog::info("allocated: {}", inst.repr());
    }
    else { // if dest_addr is not on sram, initialize an entry
        _acc_spad.reserve(inst.dest_addr, inst.accum_spad_id, inst.size, 1);  // spdlog::info("reserve: {}", inst.repr());
        // spdlog::info("reserve, dest_addr:{:x}, spad_id:{}, size:{}", inst.dest_addr, inst.accum_spad_id, inst.size);
    }


    // Count the execution time of each instructions
    switch (inst.opcode) {
        case Opcode::GEMM:
        case Opcode::GEMM_PRELOAD:
            _gemm_count++;  // fixed GEMM block size
            break;
        case Opcode::LAYERNORM:
            _layernorm_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::RMSNORM:
            _rmsnorm_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::ROPE:
            _rope_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::SOFTMAX:
            _softmax_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::ADD:
            _add_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::MUL:
            _mul_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::GELU:
            _gelu_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::SILU:
            _silu_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::GEMV:
            _gemv_count += calculate_vector_op_iterations(inst.size);
            break;
        case Opcode::IM2COL:
            _im2col_count++;
            break;
        case Opcode::DATA_CONVERT:
        case Opcode::DUMMY:
            _dummy_count++;
            break;
        default:
            throw std::runtime_error("Not supported opcode");
            break;
    }
}


void MyCore::print_stats() {
    spdlog::info("---- Core [{}] : Stats ----", _id);
    spdlog::info("Core [{}] : Read count {}, Write count {}",
        _id, _read_count, _write_count);
    if (_id == 0) {
        spdlog::info("Core [{}] : PIM_PHeader count {}, PIM_GWrite count {}, PIM_Comp count {}, PIM_Readers count {}",
            _id, _pim_pheader_count, _pim_gwrite_count, _pim_comp_count, _pim_readers_count);
    }
    spdlog::info("Core [{}] : GEMM count {}, LayerNorm count {}, RMSNorm count {}, RoPE count {}, Softmax count {}, Add count {}, Mul count {}, Gelu count {}, SiLU count {}, GEMV count {}",
        _id, _gemm_count, _layernorm_count, _rmsnorm_count, _rope_count, _softmax_count, _add_count, _mul_count, _gelu_count, _silu_count, _gemv_count);
    spdlog::info("Core [{}] : GEMM cycle {}, LayerNorm cycle {}, RMSNorm cycle {}, RoPE cycle {}, Softmax cycle {}, Add cycle {}, Mul cycle {}, Gelu cycle {}, SiLU cycle {}, GEMV cycle {}",
        _id, _stat_gemm_cycle, _stat_layernorm_cycle, _stat_rmsnorm_cycle, _stat_rope_cycle, _stat_softmax_cycle, _stat_add_cycle, _stat_mul_cycle, _stat_gelu_cycle, _stat_silu_cycle, _stat_gemv_cycle);
    spdlog::info("Core [{}] : GEMM stall cycle {}, LayerNorm stall cycle {}, RMSNorm stall cycle {}, RoPE stall cycle {}, Softmax stall cycle {}, Add stall cycle {}, Mul stall cycle {}, Gelu stall cycle {}, SiLU stall cycle {}, GEMV stall cycle {}",
        _id, _compute_memory_stall_cycle, _layernorm_stall_cycle, _rmsnorm_stall_cycle, _rope_stall_cycle, _softmax_stall_cycle, _add_stall_cycle, _mul_stall_cycle, _gelu_stall_cycle, _silu_stall_cycle, _gemv_stall_cycle);
    spdlog::info("Core [{}] : Load stall cycle {}, Store stall cycle {}, Total memory stall {}, Idle cycle {}",
        _id, _load_memory_cycle, _store_memory_cycle,_stat_memory_cycle, _stat_idle_cycle);
    spdlog::info(
        "Core [{}] : Estimated workload tiles {}, Read count {}, Write count {}, Read bytes {}, Write bytes {}, Calculations {}",
        _id, _estimated_workload.tiles, _estimated_workload.memory_reads,
        _estimated_workload.memory_writes, _estimated_workload.memory_read_bytes,
        _estimated_workload.memory_write_bytes, _estimated_workload.num_calculation);
    const cycle_type accounted_cycles = _stat_compute_cycle + _stat_memory_cycle;
    const cycle_type active_memory_stall =
        _stat_memory_cycle >= _stat_idle_cycle
            ? _stat_memory_cycle - _stat_idle_cycle : 0;
    const double compute_util = accounted_cycles == 0 ? 0.0
        : static_cast<double>(_stat_compute_cycle) / accounted_cycles;
    const double memory_stall_util = accounted_cycles == 0 ? 0.0
        : static_cast<double>(active_memory_stall) / accounted_cycles;
    const double idle_util = accounted_cycles == 0 ? 0.0
        : static_cast<double>(_stat_idle_cycle) / accounted_cycles;
    spdlog::info(
        "Core [{}] : Logical timing compute {}, active memory stall {}, idle {}, accounted {}, utilization compute/stall/idle {:.6f}/{:.6f}/{:.6f}",
        _id, _stat_compute_cycle, active_memory_stall, _stat_idle_cycle,
        accounted_cycles, compute_util, memory_stall_util, idle_util);
    spdlog::info(
        "Core [{}] : Estimated timing compute {}, memory {}, idle {}, load {}, store {}",
        _id, _estimated_timing.compute, _estimated_timing.memory,
        _estimated_timing.idle, _estimated_timing.load,
        _estimated_timing.store);
    std::ofstream timing_out(
        Config::system_config.log_dir + "/core_timing.tsv",
        _id == 0 ? std::ofstream::out : std::ofstream::app);
    if (_id == 0) {
        timing_out
            << "core\ttotal_cycle\taccounted_cycle\tcompute_cycle\tactive_memory_stall\tidle_cycle"
            << "\testimated_compute\testimated_memory\testimated_idle"
            << "\tcompute_util\tmemory_stall_util\tidle_util"
            << "\tgemm_cycle\tlayernorm_cycle\trmsnorm_cycle\trope_cycle\tsoftmax_cycle"
            << "\tadd_cycle\tmul_cycle\tgelu_cycle\tsilu_cycle\tgemv_cycle"
            << "\tgemm_stall\tlayernorm_stall\trmsnorm_stall\trope_stall\tsoftmax_stall"
            << "\tadd_stall\tmul_stall\tgelu_stall\tsilu_stall\tgemv_stall\n";
    }
    timing_out << _id << '\t' << _core_cycle << '\t' << accounted_cycles
               << '\t' << _stat_compute_cycle << '\t' << active_memory_stall
               << '\t' << _stat_idle_cycle << '\t' << _estimated_timing.compute
               << '\t' << _estimated_timing.memory << '\t'
               << _estimated_timing.idle << '\t' << compute_util << '\t'
               << memory_stall_util << '\t' << idle_util << '\t'
               << _stat_gemm_cycle << '\t' << _stat_layernorm_cycle << '\t'
               << _stat_rmsnorm_cycle << '\t' << _stat_rope_cycle << '\t'
               << _stat_softmax_cycle << '\t' << _stat_add_cycle << '\t'
               << _stat_mul_cycle << '\t' << _stat_gelu_cycle << '\t'
               << _stat_silu_cycle << '\t' << _stat_gemv_cycle << '\t'
               << _compute_memory_stall_cycle << '\t' << _layernorm_stall_cycle
               << '\t' << _rmsnorm_stall_cycle << '\t' << _rope_stall_cycle
               << '\t' << _softmax_stall_cycle << '\t' << _add_stall_cycle
               << '\t' << _mul_stall_cycle << '\t' << _gelu_stall_cycle
               << '\t' << _silu_stall_cycle << '\t' << _gemv_stall_cycle
               << '\n';
    spdlog::info("Core [{}] : Total cycle: {}", _id, _core_cycle);
}

void MyCore::apply_estimated_workload(const ProportionalWorkloadStat& workload) {
    _estimated_workload += workload;
    if (!_stat.empty()) {
        _stat.back().num_calculations += workload.num_calculation;
    }

    _read_count += workload.memory_reads;
    _write_count += workload.memory_writes;
    _pim_pheader_count += workload.pim_pheader;
    _pim_gwrite_count += workload.pim_gwrite;
    _pim_comp_count += workload.pim_comp;
    _pim_readers_count += workload.pim_readres;

    _gemm_count += workload.gemm;
    _gemv_count += workload.gemv;
    _layernorm_count += workload.layernorm;
    _rmsnorm_count += workload.rmsnorm;
    _rope_count += workload.rope;
    _softmax_count += workload.softmax;
    _add_count += workload.add;
    _mul_count += workload.mul;
    _gelu_count += workload.gelu;
    _silu_count += workload.silu;
    _im2col_count += workload.im2col;
    _dummy_count += workload.dummy;
}

MyCore::TimingBreakdown MyCore::timing_snapshot() const {
    TimingBreakdown timing;
    timing.cycle = _core_cycle;
    timing.compute = _stat_compute_cycle;
    timing.memory = _stat_memory_cycle;
    timing.idle = _stat_idle_cycle;
    timing.load = _load_memory_cycle;
    timing.store = _store_memory_cycle;
    timing.op_compute = {
        _stat_gemm_cycle, _stat_layernorm_cycle, _stat_rmsnorm_cycle,
        _stat_rope_cycle, _stat_softmax_cycle, _stat_add_cycle,
        _stat_mul_cycle, _stat_gelu_cycle, _stat_silu_cycle,
        _stat_gemv_cycle};
    timing.op_stall = {
        _compute_memory_stall_cycle, _layernorm_stall_cycle,
        _rmsnorm_stall_cycle, _rope_stall_cycle, _softmax_stall_cycle,
        _add_stall_cycle, _mul_stall_cycle, _gelu_stall_cycle,
        _silu_stall_cycle, _gemv_stall_cycle};
    return timing;
}

void MyCore::apply_decode_pruning_timing(const TimingBreakdown& timing) {
    _stat_compute_cycle += timing.compute;
    _stat_memory_cycle += timing.memory;
    _stat_idle_cycle += timing.idle;
    _load_memory_cycle += timing.load;
    _store_memory_cycle += timing.store;

    _stat_gemm_cycle += timing.op_compute[0];
    _stat_layernorm_cycle += timing.op_compute[1];
    _stat_rmsnorm_cycle += timing.op_compute[2];
    _stat_rope_cycle += timing.op_compute[3];
    _stat_softmax_cycle += timing.op_compute[4];
    _stat_add_cycle += timing.op_compute[5];
    _stat_mul_cycle += timing.op_compute[6];
    _stat_gelu_cycle += timing.op_compute[7];
    _stat_silu_cycle += timing.op_compute[8];
    _stat_gemv_cycle += timing.op_compute[9];

    _compute_memory_stall_cycle += timing.op_stall[0];
    _layernorm_stall_cycle += timing.op_stall[1];
    _rmsnorm_stall_cycle += timing.op_stall[2];
    _rope_stall_cycle += timing.op_stall[3];
    _softmax_stall_cycle += timing.op_stall[4];
    _add_stall_cycle += timing.op_stall[5];
    _mul_stall_cycle += timing.op_stall[6];
    _gelu_stall_cycle += timing.op_stall[7];
    _silu_stall_cycle += timing.op_stall[8];
    _gemv_stall_cycle += timing.op_stall[9];

    _estimated_timing.compute += timing.compute;
    _estimated_timing.memory += timing.memory;
    _estimated_timing.idle += timing.idle;
    _estimated_timing.load += timing.load;
    _estimated_timing.store += timing.store;
    _estimated_timing.cycle += timing.cycle;
    for (size_t op = 0; op < kTimingOpCount; ++op) {
        _estimated_timing.op_compute[op] += timing.op_compute[op];
        _estimated_timing.op_stall[op] += timing.op_stall[op];
    }
}

void MyCore::begin_proportional_timing_sampling() {
    _proportional_timing_baseline = timing_snapshot();
    _proportional_timing_sample_complete = false;
}

void MyCore::end_proportional_timing_sampling() {
    _proportional_timing_sample_end = timing_snapshot();
    _proportional_timing_sample_complete = true;
}

MyCore::TimingBreakdown MyCore::apply_estimated_timing(
    cycle_type skipped_work_cycles, cycle_type global_wait_cycles) {
    const TimingBreakdown live = timing_snapshot();
    const TimingBreakdown current = _proportional_timing_sample_complete
        ? _proportional_timing_sample_end : timing_snapshot();
    const cycle_type sampled_cycles =
        current.cycle - _proportional_timing_baseline.cycle;
    const auto delta = [](cycle_type current_value,
                          cycle_type baseline_value) -> cycle_type {
        return current_value >= baseline_value
            ? current_value - baseline_value : 0;
    };
    const auto scale = [](cycle_type measured, cycle_type target,
                          cycle_type sampled) -> cycle_type {
        if (measured == 0 || sampled == 0 || target == 0) return 0;
        return static_cast<cycle_type>(std::llround(
            static_cast<long double>(measured) * target / sampled));
    };

    const cycle_type sampled_compute =
        delta(current.compute, _proportional_timing_baseline.compute);
    const cycle_type sampled_memory =
        delta(current.memory, _proportional_timing_baseline.memory);
    const cycle_type sampled_idle = std::min<cycle_type>(
        sampled_memory,
        delta(current.idle, _proportional_timing_baseline.idle));
    const cycle_type sampled_active_memory = sampled_memory - sampled_idle;
    const cycle_type sampled_accounted =
        sampled_compute + sampled_active_memory + sampled_idle;
    const cycle_type post_sample_cycles =
        _proportional_timing_sample_complete
            ? delta(live.cycle, current.cycle) : 0;
    const cycle_type post_sample_idle =
        _proportional_timing_sample_complete
            ? std::min<cycle_type>(
                post_sample_cycles, delta(live.idle, current.idle))
            : post_sample_cycles;
    // Proportional pruning removes complete interior work units.  Boundary
    // idle after the sample is already simulated, but idle observed inside
    // the sampling window belongs to those repeated work units and must be
    // scaled along with compute and active-memory time.
    // The per-core predicted finish spread includes outstanding memory work,
    // so it cannot safely be replayed as idle. Boundary idle is already
    // present in the fully simulated tail below.
    const cycle_type estimated_global_idle = 0;
    const cycle_type estimated_global_active = global_wait_cycles;

    TimingBreakdown estimated;
    estimated.cycle = skipped_work_cycles + global_wait_cycles;
    if (sampled_accounted > 0) {
        estimated.compute = std::min<cycle_type>(
            skipped_work_cycles,
            scale(sampled_compute, skipped_work_cycles, sampled_accounted));
        const cycle_type remaining_after_compute =
            skipped_work_cycles - estimated.compute;
        estimated.idle = estimated_global_idle +
            std::min<cycle_type>(
                remaining_after_compute,
                scale(sampled_idle, skipped_work_cycles,
                      sampled_accounted));
    } else {
        // No usable sample: conservatively classify predicted work as an
        // active memory stall.
        estimated.compute = 0;
        estimated.idle = estimated_global_idle;
    }

    const cycle_type estimated_work_idle =
        estimated.idle - estimated_global_idle;
    const cycle_type estimated_work_active_memory =
        skipped_work_cycles - estimated.compute - estimated_work_idle;
    const cycle_type estimated_active_memory =
        estimated_work_active_memory + estimated_global_active;
    estimated.memory =
        estimated_active_memory + estimated_work_idle +
        estimated_global_idle;

    const cycle_type sampled_load = std::min<cycle_type>(
        sampled_active_memory,
        delta(current.load, _proportional_timing_baseline.load));
    estimated.load = sampled_active_memory == 0 ? 0 :
        std::min<cycle_type>(
            estimated_work_active_memory,
            scale(sampled_load, estimated_work_active_memory,
                  sampled_active_memory));
    const cycle_type estimated_active_store =
        estimated_work_active_memory - estimated.load;
    estimated.store =
        estimated_active_store + estimated_global_active +
        estimated_work_idle + estimated_global_idle;

    for (size_t op = 0; op < kTimingOpCount; ++op) {
        estimated.op_compute[op] = scale(
            delta(current.op_compute[op],
                  _proportional_timing_baseline.op_compute[op]),
            estimated.compute, sampled_compute);
        estimated.op_stall[op] = scale(
            delta(current.op_stall[op],
                  _proportional_timing_baseline.op_stall[op]),
            estimated.load, sampled_load);
    }

    cycle_type attributed_stalls = 0;
    for (cycle_type op_stall : estimated.op_stall) {
        attributed_stalls += op_stall;
    }
    if (attributed_stalls > estimated.load && attributed_stalls > 0) {
        cycle_type excess = attributed_stalls - estimated.load;
        while (excess > 0) {
            auto largest = std::max_element(
                estimated.op_stall.begin(), estimated.op_stall.end());
            if (largest == estimated.op_stall.end() || *largest == 0) break;
            const cycle_type reduction = std::min<cycle_type>(*largest, excess);
            *largest -= reduction;
            excess -= reduction;
        }
    }

    _stat_compute_cycle += estimated.compute;
    _stat_memory_cycle += estimated.memory;
    _stat_idle_cycle += estimated.idle;
    _load_memory_cycle += estimated.load;
    _store_memory_cycle += estimated.store;

    _stat_gemm_cycle += estimated.op_compute[0];
    _stat_layernorm_cycle += estimated.op_compute[1];
    _stat_rmsnorm_cycle += estimated.op_compute[2];
    _stat_rope_cycle += estimated.op_compute[3];
    _stat_softmax_cycle += estimated.op_compute[4];
    _stat_add_cycle += estimated.op_compute[5];
    _stat_mul_cycle += estimated.op_compute[6];
    _stat_gelu_cycle += estimated.op_compute[7];
    _stat_silu_cycle += estimated.op_compute[8];
    _stat_gemv_cycle += estimated.op_compute[9];

    _compute_memory_stall_cycle += estimated.op_stall[0];
    _layernorm_stall_cycle += estimated.op_stall[1];
    _rmsnorm_stall_cycle += estimated.op_stall[2];
    _rope_stall_cycle += estimated.op_stall[3];
    _softmax_stall_cycle += estimated.op_stall[4];
    _add_stall_cycle += estimated.op_stall[5];
    _mul_stall_cycle += estimated.op_stall[6];
    _gelu_stall_cycle += estimated.op_stall[7];
    _silu_stall_cycle += estimated.op_stall[8];
    _gemv_stall_cycle += estimated.op_stall[9];

    _estimated_timing.compute += estimated.compute;
    _estimated_timing.memory += estimated.memory;
    _estimated_timing.idle += estimated.idle;
    _estimated_timing.load += estimated.load;
    _estimated_timing.store += estimated.store;
    _estimated_timing.cycle += estimated.cycle;
    for (size_t op = 0; op < kTimingOpCount; ++op) {
        _estimated_timing.op_compute[op] += estimated.op_compute[op];
        _estimated_timing.op_stall[op] += estimated.op_stall[op];
    }
    spdlog::info(
        "Core [{}] timing compensation: sampled {}, post-sample/idle {}/{}, skipped work {}, global wait {} active/idle {}/{}, estimated compute/active-memory/idle {}/{}/{}",
        _id, sampled_cycles, post_sample_cycles, post_sample_idle,
        skipped_work_cycles, global_wait_cycles,
        estimated_global_active, estimated_global_idle,
        estimated.compute, estimated.memory - estimated.idle,
        estimated.idle);
    return estimated;
}

void MyCore::log() {}


void MyCore::update_stats() {
    if (!_compute_pipeline.empty()) {
        auto parent_tile = _compute_pipeline.front().parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        parent_tile->stat.compute_cycles++;
        _stat.back().num_calculations += _config.core_height * _config.core_width ;   // 128 * 8 * 2;  // apply systolic array count
    }
    for (auto &vector_pipeline : _vector_pipelines) {
        if (!vector_pipeline.empty()) {
            auto parent_tile = vector_pipeline.front().parent_tile.lock();
            if (parent_tile == nullptr) {
                assert(0);
            }
            parent_tile->stat.compute_cycles++;
            _stat.back().num_calculations += _config.vector_core_width;  // apply vector size
        }
    }

    bool is_idle = _compute_pipeline.empty();
    for (auto &vector_pipeline : _vector_pipelines) {
        is_idle = is_idle && vector_pipeline.empty();
    }
    if (is_idle) {
        _stat_memory_cycle++;
        if (_ex_inst_queue.empty()) {
            _store_memory_cycle++;
        } else {
            _load_memory_cycle++;
            switch (_ex_inst_queue.front().opcode) {
                case Opcode::GEMM:
                case Opcode::GEMM_PRELOAD:
                    _compute_memory_stall_cycle++;
                    break;
                case Opcode::LAYERNORM:
                    _layernorm_stall_cycle++;
                    break;
                case Opcode::RMSNORM:
                    _rmsnorm_stall_cycle++;
                    break;
                case Opcode::ROPE:
                    _rope_stall_cycle++;
                    break;
                case Opcode::SOFTMAX:
                    _softmax_stall_cycle++;
                    break;
                case Opcode::ADD:
                    _add_stall_cycle++;
                    break;
                case Opcode::MUL:
                    _mul_stall_cycle++;
                    break;
                case Opcode::GELU:
                    _gelu_stall_cycle++;
                    break;
                case Opcode::SILU:
                    _silu_stall_cycle++;
                    break;
                case Opcode::GEMV:
                    _gemv_stall_cycle++;
                    break;
                default:
                    break;
            }
        }
    } else if (!_compute_pipeline.empty()) {  // GEMM Computation
        _stat_compute_cycle++;
        _stat_gemm_cycle++;
    } else {
        _stat_compute_cycle++;
        // } else if (!_vector_pipeline.empty()) {
        // when element in vector pipeline
        for (auto &vector_pipeline : _vector_pipelines) {
            switch (vector_pipeline.front().opcode) {
                case Opcode::LAYERNORM:
                    _stat_layernorm_cycle++;
                    break;
                case Opcode::RMSNORM:
                    _stat_rmsnorm_cycle++;
                    break;
                case Opcode::ROPE:
                    _stat_rope_cycle++;
                    break;
                case Opcode::SOFTMAX:
                    _stat_softmax_cycle++;
                    break;
                case Opcode::ADD:
                    _stat_add_cycle++;
                    break;
                case Opcode::MUL:
                    _stat_mul_cycle++;
                    break;
                case Opcode::GELU:
                    _stat_gelu_cycle++;
                    break;
                case Opcode::SILU:
                    _stat_silu_cycle++;
                    break;
                case Opcode::GEMV:
                    _stat_gemv_cycle++;
                    break;
                default:
                    break;
            }
        }
    }
    if (!running()) {
        _stat_idle_cycle++;
    }
}


void MyCore::pim_ld_queue_cycle() {
    /* LD instruction queue */
    // todo: ld_queue.cycle();
    while (!_ld_inst_queue_for_pim.empty()) {
        Instruction &front = _ld_inst_queue_for_pim.front();
        // spdlog::info("{}", front.repr());
        if (front.opcode == Opcode::PIM_HEADER || front.opcode == Opcode::PIM_GWRITE ||
            front.opcode == Opcode::PIM_COMP || front.opcode == Opcode::PIM_COMP_HASH ||
            front.opcode == Opcode::PIM_READRES || front.opcode == Opcode::PIM_COMPS_READRES) {
            Sram *buffer;
            int buffer_id;
            if (front.dest_addr >= ACCUM_SPAD_BASE) {
                buffer = &_pim_acc_spad;
                buffer_id = front.accum_spad_id;
            } else {
                buffer = &_pim_spad;
                buffer_id = front.spad_id;
            }
            ast(!front.src_addrs.empty());
            auto mem_request = TransToMemoryAccess(
                front, MyAddressAllocator::dram_burst_size, _id, _core_cycle, buffer_id, StagePlatform::PIM);

            if (front.opcode == Opcode::PIM_READRES || front.opcode == Opcode::PIM_COMPS_READRES)
                buffer->reserve(front.dest_addr, buffer_id, front.size, 1);

            push_memory_request(std::move(mem_request));
            _ld_inst_queue_for_pim.pop();

            } else {
                assert(0);
            }
    }
}


void MyCore::pim_st_queue_cycle() {
    /* ST instruction queue */
    // todo: st_queue.cycle();
    if (!_st_inst_queue_for_pim.empty()) {
        Instruction &front = _st_inst_queue_for_pim.front();
        Sram *buffer;
        int buffer_id;
        if (front.dest_addr >= ACCUM_SPAD_BASE) {
            buffer = &_pim_acc_spad;
            buffer_id = front.accum_spad_id;
        } else {
            buffer = &_pim_spad;
            buffer_id = front.spad_id;
        }
        // spdlog::info("{}", front.repr());
        if (buffer->check_hit(front.dest_addr, buffer_id) &&
            (front.opcode == Opcode::MOVOUT || front.opcode == Opcode::MOVOUT_POOL)) {
            auto accesses = MemoryAccess::from_instruction(
                front, generate_mem_access_id(), MyAddressAllocator::dram_burst_size, MemoryAccessType::WRITE,
                true, _id, _core_cycle, buffer_id, StagePlatform::PIM);
            if (auto tile = front.parent_tile.lock()) {
                tile->remaining_accum_io += accesses.size() - 1;
                tile->stat.memory_writes += accesses.size() * AddressConfig::alignment;
            } else {
                assert(0);
            }
            for (auto& access : accesses) {
                push_memory_request(std::move(access));
                _waiting_write_reqs++;
            }
            _st_inst_queue_for_pim.pop();
            }
    }
}


void MyCore::pim_ex_queue_cycle() {
    /* EX instruction queue */
    if (!_ex_inst_queue_for_pim.empty()) {
        Instruction ready_inst = _ex_inst_queue_for_pim.front();
        if (pim_can_issue_compute(ready_inst)) {
            _ex_inst_queue_for_pim.pop();
        } else {
            ready_inst = Instruction{.valid = false};
        }

        if (ready_inst.valid)
            pim_issue_ex_inst(ready_inst);
        else {
            /* Update memory stall stat */
        }
    }
}


void MyCore::pim_issue_ex_inst(Instruction inst) {
    // spdlog::info("cycle:{}, {}", _core_cycle, inst.repr());
    if (inst.opcode == Opcode::GEMM || inst.opcode == Opcode::GEMM_PRELOAD) {
        // xxx: not yet for pim.
        assert(0);
        auto parent_tile = inst.parent_tile.lock();
        if (parent_tile == nullptr) {
            assert(0);
        }
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        parent_tile->stat.num_calculation += inst.tile_m * inst.tile_n * inst.tile_k;

        if (inst.opcode == Opcode::GEMM_PRELOAD) {
            _stat_systolic_preload_issue_count++;
        }
        if (!_compute_pipeline.empty()) {
            /* Preload can be hided */
            uint32_t offset = _compute_pipeline.back().size;
            // xxx why 4?
            // maybe pushing to the systolic array input queue. 4 cycles to start?
            offset = MAX(offset, 4);
            if (inst.opcode == Opcode::GEMM_PRELOAD) {
                // State mul-pre
                parent_tile->stat.weight_load_cycles += _config.core_height;
                offset = _config.core_height;
            }
            inst.start_cycle = _compute_pipeline.back().start_cycle + offset;
        } else {
            inst.start_cycle = _core_cycle;
            /* Preload weight to systolic array*/
            if (inst.opcode == Opcode::GEMM_PRELOAD) {
                /* Weight preload  from buffer latecny + WEight preload
                 * latency */
                inst.start_cycle += _config.core_height + _config.core_height - 1;
            }
        }

        inst.finish_cycle = inst.start_cycle + get_inst_compute_cycles(inst);
        // spdlog::info("finish_cycle: {}", inst.finish_cycle);
        _compute_pipeline.push(inst);
        _stat_systolic_inst_issue_count++;
    } else if (inst.opcode == Opcode::COMP || inst.opcode == Opcode::IM2COL ||
               inst.opcode == Opcode::LAYERNORM || inst.opcode == Opcode::RMSNORM || inst.opcode == Opcode::ROPE || inst.opcode == Opcode::SOFTMAX ||
               inst.opcode == Opcode::ADD || inst.opcode == Opcode::MUL || inst.opcode == Opcode::GELU || inst.opcode == Opcode::SILU ||
               inst.opcode == Opcode::DATA_CONVERT || inst.opcode == Opcode::DUMMY) {  // vector unit compute
        // spdlog::info("COMPUTE Start cycle: {} inst:{}", _core_cycle, inst.repr());
        std::queue<Instruction> *least_filled_vpu;
        cycle_type finish_cycle = std::numeric_limits<uint64_t>::max();
        for (auto &vector_pipeline : _vector_pipelines) {
            if (vector_pipeline.empty()) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
                break;
            }
            if (vector_pipeline.back().finish_cycle < finish_cycle) {
                least_filled_vpu = &vector_pipeline;
                finish_cycle = _core_cycle;
            }
        }
        inst.start_cycle = finish_cycle;
        inst.finish_cycle = inst.start_cycle + get_vector_compute_cycles(inst);
        least_filled_vpu->push(inst);

        {
            // if (!_vector_pipeline.empty()) {
            //     // loading latency
            //     inst.start_cycle = _vector_pipeline.back().finish_cycle + 1;
            // } else {
            //     inst.start_cycle = _core_cycle;
            // }
            // inst.finish_cycle = inst.start_cycle + get_vector_compute_cycles(inst);
            // _vector_pipeline.push(inst);
        }
    }

    // if dest_addr is on sram, count up. -> wait for _compute_pipeline to
    // finish calculation
    if (_pim_acc_spad.check_allocated(inst.dest_addr, inst.accum_spad_id)) {
        // spdlog::info("allocated: {}", inst.repr());
        _pim_acc_spad.count_up(inst.dest_addr, inst.accum_spad_id);
    }
    // if dest_addr is not on sram, initialize. -> wait for
    // _compute_pipeline to finish calculation
    else {
        // spdlog::info("reserve: {}", inst.repr());
        // spdlog::info("reserve, dest_addr:{:x}, spad_id:{}, size:{}", inst.dest_addr,
        //  inst.accum_spad_id, inst.size);
        _pim_acc_spad.reserve(inst.dest_addr, inst.accum_spad_id, inst.size, 1);
    }
}
