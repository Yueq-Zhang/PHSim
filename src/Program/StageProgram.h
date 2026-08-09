#pragma once

#include <vector>
#include "../src/Client/BatchedRequest.h"
#include "../common_function.hpp"
#include "../Model/Model.h"
#include "../operations/Operation.h"
#include "../tensor/BTensor.h"

class StageProgram {
   public:
    StageProgram(std::shared_ptr<Model> model, Ptr<BatchedRequest> batched_request, StagePlatform stage_type, Stage stage);
    StageProgram(Ops test_single_op_type, std::shared_ptr<BatchedRequest> batched_request, Stage stage);  // for the single op test
    StageProgram(std::shared_ptr<Model> model, std::string test_multi_layer_name, std::shared_ptr<BatchedRequest> batched_request, Stage stage);  // for model inference

    void init_program();
    Ptr<Operation> add_op(Ptr<Operation> op);
    std::vector<Ptr<BTensor>> get_outputs(Ptr<Operation> op, std::vector<Ptr<BTensor>> inputs);
    std::vector<Ptr<MyTensor>> get_my_outputs(Ptr<Operation> op, std::vector<Ptr<MyTensor>> inputs);

    bool check_exist_in_executable(uint32_t op_id);
    void finish_operation(uint32_t id);
    void find_executable_node(Ptr<BTensor> tensor);
    void find_executable_node(Ptr<MyTensor> tensor);
    void find_executable_node(std::vector<Ptr<MyTensor>> tensors);

    std::vector<std::shared_ptr<Operation>> get_executable_operations() {
        return _executable_operations;
    }
    bool check_finish();
    std::vector<OperationStat> list_operation_stat();
    void finish_operation_tile(Tile& tile);
    void log();

    std::string _name;
    std::string _test_multi_layer_name;
    // std::unordered_map<std::string, Ptr<MyTensor>> _stored_KVCache;

    // todo: from BatchedRequest
    std::shared_ptr<Model> _model;
    std::shared_ptr<BatchedRequest> _breq;
    std::unordered_map<uint32_t, Ptr<Operation>> _op_map;
    std::vector<std::shared_ptr<Operation>> _executable_operations;

    // Sub-batch interleaving
    StagePlatform _stage_platform;
    Stage _stage;

    void init_program_single_op(Ops test_single_op_type);
    void init_program_multi_layer();
    Ptr<Operation> create_norm_op(std::string name, int layer, std::string block_type);

    // Layer Block
    std::vector<Ptr<MyTensor>> test_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer);
    std::vector<Ptr<MyTensor>> test_attn_block(std::vector<Ptr<MyTensor>> inputs, int layer, InferRequest* req);
    std::vector<Ptr<MyTensor>> test_attn_block(std::vector<Ptr<MyTensor>> inputs, int layer, std::vector<std::shared_ptr<InferRequest>> reqs);

    std::vector<Ptr<MyTensor>> test_attention_block(std::vector<Ptr<MyTensor>> inputs);
    std::vector<Ptr<MyTensor>> test_decode_stage_atten(std::vector<Ptr<MyTensor>> inputs, int layer);
    std::vector<Ptr<MyTensor>> test_decode_stage_ffn(std::vector<Ptr<MyTensor>> inputs, int layer);

    std::vector<Ptr<MyTensor>> init_program_att_block(std::vector<Ptr<MyTensor>> inputs, int layer, std::vector<std::shared_ptr<InferRequest>> reqs);
    std::vector<Ptr<MyTensor>> init_program_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer);

    // new generate blocks for llama model
    std::vector<Ptr<MyTensor>> test_llama_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer);
    std::vector<Ptr<MyTensor>> test_llama_decode_stage_ffn(std::vector<Ptr<MyTensor>> inputs, int layer);
    std::vector<Ptr<MyTensor>> test_llama_attn_block(std::vector<Ptr<MyTensor>> inputs, int layer);
    std::vector<Ptr<MyTensor>> test_llama_decode_stage_atten(std::vector<Ptr<MyTensor>> inputs, int layer);
};
