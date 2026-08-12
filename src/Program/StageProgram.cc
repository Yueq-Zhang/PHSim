#include "StageProgram.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "../Model/Model.h"
#include "../tensor/BTensor.h"
#include "../tensor/MyTensor.hpp"
#include "fmt/format.h"

#include "../operations/PIMGEMV.h"
#include "../operations/GEMV.h"

#include "../operations/Add.h"
#include "../operations/Mul.h"
#include "../operations/Gelu.h"
#include "../operations/SiLU.h"
#include "../operations/LayerNorm.h"
#include "../operations/RMSNorm.h"
#include "../operations/GEMM.h"
#include "../operations/Softmax.h"
#include "../operations/DataConvert.h"

namespace {
uint32_t model_kv_cache_width() {
    return Config::system_config.model_n_kv_head *
           (Config::system_config.model_n_embd / Config::system_config.model_n_head);
}
}


StageProgram::StageProgram(Ptr<Model> model, Ptr<BatchedRequest> batched_request,
                           StagePlatform stage_platform, Stage stage,
                           DramDataContainer* data_container):
    _name(stagePlatformToString(stage_platform) + "_stage_" + stageToString(stage)),
    _model(std::move(model)),
    _breq(std::move(batched_request)),
    _stage_platform(stage_platform),
    _stage(stage),
    _data_container(data_container) {
    this->init_program();
}

StageProgram::StageProgram(Ops test_single_op_type,
                           std::shared_ptr<BatchedRequest> batched_request,
                           Stage stage, DramDataContainer* data_container):
    _model(nullptr),
    _breq(std::move(batched_request)),
    _stage_platform(StagePlatform::PIM),
    _stage(stage),
    _data_container(data_container) {
    this->init_program_single_op(test_single_op_type);
}

StageProgram::StageProgram(Ptr<Model> model, std::string test_multi_layer_name,
                           std::shared_ptr<BatchedRequest> batched_request,
                           Stage stage, DramDataContainer* data_container):
    _model(std::move(model)),
    _test_multi_layer_name(std::move(test_multi_layer_name)),
    _stage_platform(StagePlatform::PIM),
    _breq(std::move(batched_request)),
    _stage(stage),
    _data_container(data_container) {
    this->init_program_multi_layer();
}

Ptr<Operation> StageProgram::create_norm_op(std::string name, int layer, std::string block_type) {
    if (is_llama_model_name(Config::system_config.model_name)) {
        auto params = _model->get_my_params(layer, block_type, OperationType::RMSNorm);
        name = name_gen(name, OperationType::RMSNorm);
        return add_op(std::make_shared<RMSNorm>(std::move(name), params));
    }
    auto params = _model->get_my_params(layer, block_type, OperationType::LayerNorm);
    name = name_gen(name, OperationType::LayerNorm);
    return add_op(std::make_shared<LayerNorm>(std::move(name), params));
}

void StageProgram::init_program() {
    assert(_stage != Stage::Finish);
    assert(!_breq->_reqs.empty());

    if (_stage == Stage::Prefill) {
        MyAddressAllocator::activation_malloc();  // Input Activation Tensor
        uint32_t batch_size = _breq->_reqs.size();

        std::vector<Ptr<MyTensor>> inputs_attn;
        Ptr<MyTensor> input_attn;
        for (const auto& req : _breq->_reqs) {
            uint32_t Lin = req->input_size;
            uint32_t Demb = Config::system_config.model_n_embd;
            std::vector<uint32_t> input_dim_attn{Lin, Demb};
            input_attn = std::make_shared<MyTensor>("input", input_dim_attn, TensorType::ACT, true);
            inputs_attn.push_back(input_attn);
        }

        for (int i = 0; i < Config::system_config.model_n_layer; ++i) {
            auto inputs_ffn = init_program_att_block(inputs_attn, i, _breq->_reqs);
            inputs_attn = init_program_ffn_block(inputs_ffn, i);
            MyAddressAllocator::activation_refresh();  // Activation Refresh
        }
        find_executable_node(input_attn);
    }
    else if (_stage == Stage::Decode or _stage == Stage::NPU_Decode) {
        uint32_t batch_size = _breq->_reqs.size();

        std::vector<Ptr<MyTensor>> inputs_attn;
        Ptr<MyTensor> input_attn;
        for (const auto& req : _breq->_reqs) {
            uint32_t Demb = Config::system_config.model_n_embd;
            std::vector<uint32_t> input_dim_attn{1, Demb};
            input_attn = std::make_shared<MyTensor>("input", input_dim_attn, TensorType::ACT, true);
            inputs_attn.push_back(input_attn);
        }

        for (int i = 0; i < Config::system_config.model_n_layer; ++i) {
            auto inputs_ffn = init_program_att_block(inputs_attn, i, _breq->_reqs);
            inputs_attn = init_program_ffn_block(inputs_ffn, i);
            MyAddressAllocator::activation_refresh();  // Activation Refresh
        }
        find_executable_node(input_attn);
        /*
        for (const auto& req : _breq->_reqs) {
            uint32_t Lout = req->output_size;
            uint32_t Demb = Config::system_config.model_n_embd;
            std::vector<uint32_t> input_dim_attn{1, Demb};
            auto input_attn = std::make_shared<MyTensor>("input", input_dim_attn, TensorType::ACT, true);
            std::vector<Ptr<MyTensor>> inputs_attn{input_attn};
            for (int t=0; t<Lout; t++) {}
            for (int i = 0; i < Config::system_config.model_n_layer; ++i) {
                auto inputs_ffn = init_program_att_block(inputs_attn, i, _breq->_reqs);
                inputs_attn = init_program_ffn_block(inputs_ffn, i);
                MyAddressAllocator::activation_refresh();
            }
            find_executable_node(input_attn);
        }
        */
    }
}


void StageProgram::init_program_single_op(Ops test_single_op_type) {
    assert(_stage == Stage::Single_test);
    spdlog::info(">>>>>> Initialize Single Layer Model Program <<<<<<");

    // the type of the operation can be required
    uint32_t batch_size = _breq->_reqs.size();

    uint32_t Lin = _breq->get_num_rows();
    Lin = 256;
    uint32_t Demb = Config::system_config.model_n_embd;
    uint32_t num_heads = Config::system_config.model_n_head;

    if (test_single_op_type == Ops::GEMM) {
        std::vector<uint32_t> weight_dim = {static_cast<uint32_t>(Demb * 1.0), Demb};
        auto my_bias = std::make_shared<MyTensor>("GEMM_bias", std::vector<uint32_t>{weight_dim[1]}, TensorType::WGT, true);
        auto my_weight = std::make_shared<MyTensor>("GEMM_weight", weight_dim, TensorType::WGT, true);

        // Missing DataContainer bursts represent zero-initialized bias/weight.

        MyAddressAllocator::activation_malloc();
        // generate activation for multi batches
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, static_cast<uint32_t>(Demb * 1.0)} ;
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            // Missing DataContainer bursts represent zero-initialized activation.

            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<GEMM>("GEMM_op_test", std::vector<Ptr<MyTensor>>{my_weight, my_bias}));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs, TensorType::VCache);
        find_executable_node(my_inputs[0]); // add the single op for execution
    }
    else if (test_single_op_type == Ops::GEMM_Att) {
        auto test_type = TensorType::VCache;
        if (test_type == TensorType::KCache) {
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs_Q;
            std::vector<Ptr<MyTensor>> my_inputs_K;

            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> Q_dim = {req->input_size, Demb};
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                auto my_Q = std::make_shared<MyTensor>("Q", Q_dim, TensorType::ACT, true);
                auto my_K = std::make_shared<MyTensor>("K", KVCache_dim, TensorType::KCache, true);
                // Missing DataContainer bursts represent zero-initialized Q/K data.
                my_inputs_Q.push_back(my_Q);
                my_inputs_K.push_back(my_K);
            }

            spdlog::info("QKT Operation Test");
            auto my_single_op = add_op(std::make_shared<GEMM>("GEMM_atten_QKT_test", my_inputs_K));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_Q, TensorType::ACT);
            find_executable_node(my_inputs_Q[0]);
        }
        else if (test_type == TensorType::VCache) {
            MyAddressAllocator::activation_malloc();  // S is Activation, V is V Cache
            std::vector<Ptr<MyTensor>> my_inputs_S;
            std::vector<Ptr<MyTensor>> my_inputs_V;

            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                std::vector<uint32_t> S_dim = {MyAddressAllocator::h, req->input_size, req->input_size};
                auto my_S = std::make_shared<MyTensor>("S", S_dim, TensorType::ACT, true);
                auto my_V = std::make_shared<MyTensor>("V", KVCache_dim, TensorType::VCache, true);
                if (_data_container != nullptr) {
                    // Score tensor
                    std::vector<uint8_t> score_data(my_S->get_total_size(), 1);
                    my_S->append_data_into_container(*_data_container, score_data);
                    // Value tensor
                    std::vector<uint8_t> value_data(my_V->get_total_size(), 1);
                    my_V->append_data_into_container(*_data_container, value_data);
                }
                my_inputs_S.push_back(my_S);
                my_inputs_V.push_back(my_V);
            }
            spdlog::info("SV Operation Test");
            auto my_single_op = add_op(std::make_shared<GEMM>("GEMM_atten_SV_test", my_inputs_V));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_S, TensorType::ACT);
            find_executable_node(my_inputs_S[0]);
        }
        else {
            assert(false);
        }
    }
    else if (test_single_op_type == Ops::RMSNorm) {
        std::vector<uint32_t> weight_dim = {Demb};
        auto gamma = std::make_shared<MyTensor>("rmsnorm_gamma", weight_dim, TensorType::WGT, true);
        MyAddressAllocator::activation_malloc();

        std::vector<Ptr<MyTensor>> my_inputs;
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<RMSNorm>("rmsnorm_op_test", std::vector<Ptr<MyTensor>>{gamma}));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::LayerNorm) {
        // weight
        std::vector<uint32_t> weight_dim = {Demb};  // gamma&beta tensor dimension
        auto gamma = std::make_shared<MyTensor>("layernorm_gamma", weight_dim, TensorType::WGT, true);
        auto beta = std::make_shared<MyTensor>("layernorm_beta", weight_dim, TensorType::WGT, true);
        MyAddressAllocator::activation_malloc();
        // Activation for multi-batch
        std::vector<Ptr<MyTensor>> my_inputs;
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<LayerNorm>("layernorm_op_test", std::vector<Ptr<MyTensor>>{gamma, beta}));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);
        find_executable_node(my_inputs[0]); // add the single op for execution
    }
    else if (test_single_op_type == Ops::Gelu) {
        MyAddressAllocator::activation_malloc();
        // Activation for multi-batch
        std::vector<Ptr<MyTensor>> my_inputs;
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input = std::make_shared<MyTensor>("gelu_input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<Gelu>("gelu_op_test"));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);

        spdlog::info("Test_gelu {}", my_single_op->get_name());
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::SiLU) {
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs;
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb};
            auto my_input = std::make_shared<MyTensor>("silu_input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<SiLU>("silu_op_test"));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);

        spdlog::info("Test_silu {}", my_single_op->get_name());
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::Softmax) {
        MyAddressAllocator::activation_malloc();
        // Activation for multi-batch
        std::vector<Ptr<MyTensor>> my_inputs;
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {MyAddressAllocator::h, req->input_size, req->input_size} ;
            auto my_input = std::make_shared<MyTensor>("softmax_input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<Softmax>("softmax_op_test"));
        spdlog::info("Test_softmax {}", my_single_op->get_name());
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::Add) {
        MyAddressAllocator::activation_malloc();
        // Activation for multi-batch
        std::vector<Ptr<MyTensor>> my_inputs_1;
        std::vector<Ptr<MyTensor>> my_inputs_2;

        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input1 = std::make_shared<MyTensor>("add_1", input_dim, TensorType::ACT, true);
            auto my_input2  = std::make_shared<MyTensor>("add_2", input_dim, TensorType::ACT, true);
            my_inputs_1.push_back(my_input1);
            my_inputs_2.push_back(my_input2);
        }
        std::vector<Ptr<MyTensor>> my_inputs;
        my_inputs.reserve(my_inputs_1.size() + my_inputs_2.size());
        my_inputs.insert(my_inputs.end(), my_inputs_1.begin(), my_inputs_1.end());
        my_inputs.insert(my_inputs.end(), my_inputs_2.begin(), my_inputs_2.end());

        auto my_single_op = add_op(std::make_shared<Add>("Add_op_test"));
        spdlog::info("Test_Add {}", my_single_op->get_name());
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::Mul) {
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs_1;
        std::vector<Ptr<MyTensor>> my_inputs_2;

        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb};
            auto my_input1 = std::make_shared<MyTensor>("mul_1", input_dim, TensorType::ACT, true);
            auto my_input2 = std::make_shared<MyTensor>("mul_2", input_dim, TensorType::ACT, true);
            my_inputs_1.push_back(my_input1);
            my_inputs_2.push_back(my_input2);
        }
        std::vector<Ptr<MyTensor>> my_inputs;
        my_inputs.reserve(my_inputs_1.size() + my_inputs_2.size());
        my_inputs.insert(my_inputs.end(), my_inputs_1.begin(), my_inputs_1.end());
        my_inputs.insert(my_inputs.end(), my_inputs_2.begin(), my_inputs_2.end());

        auto my_single_op = add_op(std::make_shared<Mul>("Mul_op_test"));
        spdlog::info("Test_Mul {}", my_single_op->get_name());
        auto my_outputs = my_single_op->get_my_outputs(my_inputs);
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::DataConvert) {
        AllocationScheme src_scheme = MyAddressAllocator::allocation_scheme;
        AllocationScheme dst_scheme;
        if (src_scheme == AllocationScheme::NPU) {
            dst_scheme = AllocationScheme::IANUS;
        }
        else if (src_scheme == AllocationScheme::IANUS) {
            dst_scheme = AllocationScheme::NPU;
        }
        else {
            throw std::runtime_error("DataConvert single op only supports NPU <-> IANUS");
        }
        std::vector<uint32_t> weight_dim = {128, 256};
        auto my_weight = std::make_shared<MyTensor>(
            "DataConvert_weight_src", weight_dim, TensorType::WGT, true, src_scheme);
        auto my_single_op = add_op(std::make_shared<DataConvert>(
            "DataConvert_single_op_test", src_scheme, dst_scheme));
        spdlog::info("Test_DataConvert {}", my_single_op->get_name());
        auto my_outputs = my_single_op->get_my_outputs({my_weight}, TensorType::WGT);
        find_executable_node(my_weight);
    }
    else if (test_single_op_type == Ops::GEMV) {
        bool test_KVcache_append = false;
        std::vector<uint32_t> weight_dim = {Demb, Demb};
        auto my_bias = std::make_shared<MyTensor>("GEMV_bias", std::vector<uint32_t>{weight_dim[1]}, TensorType::WGT, true);
        auto my_weight = std::make_shared<MyTensor>("GEMV_weight", weight_dim, TensorType::WGT, true);
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {1, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_single_op = add_op(std::make_shared<GEMV>("MyGEMV_op_test_Weight",  std::vector<Ptr<MyTensor>>{my_weight, my_bias}));
        spdlog::info("Test GEMV Operation {}", my_single_op->get_name());

        if (test_KVcache_append) {
            // KV Cache
            std::vector<Ptr<MyTensor>> my_caches;
            for (auto req : _breq->_reqs) {
                std::vector<uint32_t> cache_dim = {req->input_size, Demb};
                auto my_cache = std::make_shared<MyTensor>("PIMGEMV_Cache", cache_dim, TensorType::KCache, true);
                my_caches.push_back(my_cache);
            }
            auto Cache = my_single_op->kvcache_append(my_inputs, my_caches, TensorType::VCache);
        }
        else {
            auto my_outputs = my_single_op->get_my_outputs(my_inputs, TensorType::ACT);
        }
        find_executable_node(my_inputs[0]);
    }
    else if (test_single_op_type == Ops::GEMV_Att) {
        auto test_type = TensorType::WGT;
        if (test_type == TensorType::KCache) {
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs_Q;
            std::vector<Ptr<MyTensor>> my_inputs_K;

            std::vector<uint32_t> Q_dim = {1, Demb};
            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                my_inputs_Q.push_back(std::make_shared<MyTensor>("Q", Q_dim, TensorType::ACT, true));
                my_inputs_K.push_back(std::make_shared<MyTensor>("K", KVCache_dim, TensorType::KCache, true));
            }
            spdlog::info("QKT GEMV Operation Test");
            auto my_single_op = add_op(std::make_shared<GEMV>("GEMV_atten_QKT_test", my_inputs_K));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_Q, TensorType::ACT);
            find_executable_node(my_inputs_Q[0]);
        }
        else if (test_type == TensorType::VCache) {
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs_S;
            std::vector<Ptr<MyTensor>> my_inputs_V;
            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> S_dim = {MyAddressAllocator::h, 1, req->input_size};
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                my_inputs_S.push_back(std::make_shared<MyTensor>("S", S_dim, TensorType::ACT, true));
                my_inputs_V.push_back(std::make_shared<MyTensor>("V", KVCache_dim, TensorType::VCache, true));
            }
            spdlog::info("SV GEMV Operation Test");
            auto my_single_op = add_op(std::make_shared<GEMV>("GEMV_atten_SV_test", my_inputs_V));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_S, TensorType::ACT);
            find_executable_node(my_inputs_S[0]);
        }
    }
    else if (test_single_op_type == Ops::PIM_GEMV) {
        TensorType matrix_tensor_type = TensorType::WGT;
        if (matrix_tensor_type == TensorType::WGT) {
            bool test_KVcache_append = false;
            std::vector<uint32_t> weight_dim = {Demb, Demb};
            auto my_bias = std::make_shared<MyTensor>("PIMGEMV_bias", std::vector<uint32_t>{weight_dim[1]}, TensorType::WGT, true);
            auto my_weight = std::make_shared<MyTensor>("PIMGEMV_weight", weight_dim, TensorType::WGT, true);
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs;
            for (auto req : _breq->_reqs) {
                std::vector<uint32_t> input_dim = {1, Demb};
                auto my_input = std::make_shared<MyTensor>("PIMGEMV_input", input_dim, TensorType::ACT, true);
                my_inputs.push_back(my_input);
            }
            auto my_single_op = add_op(std::make_shared<PIMGEMV>("PIMGEMV_op_test_Weight",  std::vector<Ptr<MyTensor>>{my_weight, my_bias}));
            if (test_KVcache_append) {
                // Append the initialized KV cache for the single-op test.
                std::vector<Ptr<MyTensor>> my_caches;
                for (auto req : _breq->_reqs) {
                    std::vector<uint32_t> cache_dim = {req->input_size, model_kv_cache_width()};
                    auto my_cache = std::make_shared<MyTensor>("PIMGEMV_Cache", cache_dim, TensorType::KCache, true);
                    my_caches.push_back(my_cache);
                }
                auto my_outputs = my_single_op->kvcache_append(my_inputs, my_caches, TensorType::VCache);
            }
            else {
                auto my_outputs = my_single_op->get_my_outputs(my_inputs);
            }
            find_executable_node(my_inputs[0]);
        }
        else if (matrix_tensor_type == TensorType::KCache) {
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs_Q;
            std::vector<Ptr<MyTensor>> my_inputs_K;

            std::vector<uint32_t> Q_dim = {1, Demb};
            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                my_inputs_Q.push_back(std::make_shared<MyTensor>("Q", Q_dim, TensorType::ACT, true));
                my_inputs_K.push_back(std::make_shared<MyTensor>("K", KVCache_dim, TensorType::KCache, true));
            }
            spdlog::info("QKT PIM-GEMV Operation Test");
            auto my_single_op = add_op(std::make_shared<PIMGEMV>("PIM_GEMV_atten_QKT_test", my_inputs_K));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_Q, TensorType::ACT);
            find_executable_node(my_inputs_Q[0]);
        }
        else if (matrix_tensor_type == TensorType::VCache) {
            MyAddressAllocator::activation_malloc();
            std::vector<Ptr<MyTensor>> my_inputs_S;
            std::vector<Ptr<MyTensor>> my_inputs_V;

            for (const auto& req : _breq->_reqs) {
                std::vector<uint32_t> S_dim = {MyAddressAllocator::h, 1, req->input_size};
                std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
                my_inputs_S.push_back(std::make_shared<MyTensor>("S", S_dim, TensorType::ACT, true));
                my_inputs_V.push_back(std::make_shared<MyTensor>("V", KVCache_dim, TensorType::VCache, true));
            }

            spdlog::info("SV PIM-GEMV Operation Test");
            auto my_single_op = add_op(std::make_shared<PIMGEMV>("PIM_GEMV_atten_SV_test", my_inputs_V));
            auto my_outputs = my_single_op->get_my_outputs(my_inputs_S, TensorType::ACT);
            find_executable_node(my_inputs_S[0]);
        }
        else {
            throw std::logic_error("Invalid PIM GEMV operation type, may not support");
        }
    }
    else if (test_single_op_type == Ops::PIM_GEMV_QKT) {
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs_Q;
        std::vector<Ptr<MyTensor>> my_inputs_K;

        std::vector<uint32_t> Q_dim = {1, Demb};
        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
            my_inputs_Q.push_back(std::make_shared<MyTensor>("Q", Q_dim, TensorType::ACT, true));
            my_inputs_K.push_back(std::make_shared<MyTensor>("K", KVCache_dim, TensorType::KCache, true));
        }
        spdlog::info("QKT PIM-GEMV Operation Test");
        auto my_single_op = add_op(std::make_shared<PIMGEMV>("PIM_GEMV_atten_QKT_test", my_inputs_K));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs_Q, TensorType::ACT);
        find_executable_node(my_inputs_Q[0]);
    }
    else if (test_single_op_type == Ops::PIM_GEMV_SV) {
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs_S;
        std::vector<Ptr<MyTensor>> my_inputs_V;

        for (const auto& req : _breq->_reqs) {
            std::vector<uint32_t> S_dim = {MyAddressAllocator::h, 1, req->input_size};
            std::vector<uint32_t> KVCache_dim = {req->input_size, model_kv_cache_width()};
            my_inputs_S.push_back(std::make_shared<MyTensor>("S", S_dim, TensorType::ACT, true));
            my_inputs_V.push_back(std::make_shared<MyTensor>("V", KVCache_dim, TensorType::VCache, true));
        }

        spdlog::info("SV PIM-GEMV Operation Test");
        auto my_single_op = add_op(std::make_shared<PIMGEMV>("PIM_GEMV_atten_SV_test", my_inputs_V));
        auto my_outputs = my_single_op->get_my_outputs(my_inputs_S, TensorType::ACT);
        find_executable_node(my_inputs_S[0]);
    }
    else {
        throw std::runtime_error("Unsupported test single op type");
    }
}


void StageProgram::init_program_multi_layer() {
    spdlog::info(">>>>>> Initialize Multi Layer Model Program <<<<<<");
    if (_test_multi_layer_name == "ffn"){
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: ffn <<<<<<");
        MyAddressAllocator::activation_malloc();
        uint32_t Demb = Config::system_config.model_n_embd;
        // generate activation for multi batches
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_outputs = test_ffn_block(my_inputs, 0);  // layer 0
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "decode_ffn") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: decode ffn <<<<<<");
        _stage_platform = StagePlatform::PIM;
        uint32_t Demb = Config::system_config.model_n_embd;
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs;

        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {1, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        /*
        for (int i=0; i<Config::system_config.gen_request_count; i++) {
            std::vector<uint32_t> input_dim = {1, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        */
        auto outputs = test_decode_stage_ffn(my_inputs, 0);
        MyAddressAllocator::activation_refresh(); // Refresh Activation after completing transformer blocks
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "decode_attn") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: decode attn <<<<<<");
        _stage_platform = StagePlatform::PIM;
        uint32_t Demb = Config::system_config.model_n_embd;
        uint32_t num_heads = Config::system_config.model_n_head;
        MyAddressAllocator::activation_malloc();
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {1, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto inputs_ffn = test_decode_stage_atten(my_inputs, 0);
        MyAddressAllocator::activation_refresh();  // Refresh Activation after completing transformer blocks
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "attn") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: attn <<<<<<");
        uint32_t Demb = Config::system_config.model_n_embd;
        MyAddressAllocator::activation_malloc();

        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {512, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto inputs = test_attn_block(my_inputs, 0, _breq->_reqs);
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "decode" ||
             _test_multi_layer_name == "npu_decode") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: decode <<<<<<");
        _stage_platform = _test_multi_layer_name == "npu_decode"
                              ? StagePlatform::SA
                              : StagePlatform::PIM;

        uint32_t Demb = Config::system_config.model_n_embd;
        MyAddressAllocator::activation_malloc();

        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {1, Demb};
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }

        const uint32_t decode_iterations =
            Config::system_config.decode_pruning_iterations;
        std::vector<uint32_t> original_input_sizes;
        original_input_sizes.reserve(_breq->_reqs.size());
        for (const auto& req : _breq->_reqs) {
            original_input_sizes.push_back(req->input_size);
        }

        auto iteration_inputs = my_inputs;
        const bool decode_compile_pruning_enabled =
            Config::system_config.decode_pruning_compile_context;
        const uint32_t decode_sample_iterations =
            Config::system_config.decode_pruning_sample_iterations;
        for (uint32_t iteration = 0; iteration < decode_iterations;
             ++iteration) {
            const bool compile_npu_sample_fully =
                _test_multi_layer_name == "npu_decode" &&
                iteration < decode_sample_iterations;
            Config::system_config.decode_pruning_compile_context =
                decode_compile_pruning_enabled &&
                !compile_npu_sample_fully;
            spdlog::info(
                "Build Decode iteration {}/{} with KV length {}",
                iteration + 1, decode_iterations,
                _breq->_reqs.empty() ? 0 : _breq->_reqs.front()->input_size);
            auto inputs_ffn = test_decode_stage_atten(iteration_inputs, 0);
            iteration_inputs = test_decode_stage_ffn(inputs_ffn, 0);
            if (iteration + 1 < decode_iterations) {
                for (auto& req : _breq->_reqs) {
                    ++req->input_size;
                }
            }
        }
        Config::system_config.decode_pruning_compile_context =
            decode_compile_pruning_enabled;
        for (uint32_t i = 0; i < _breq->_reqs.size(); ++i) {
            _breq->_reqs[i]->input_size = original_input_sizes[i];
        }
        MyAddressAllocator::activation_refresh();  // Refresh Activation after completing transformer blocks
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "prefill") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: Prefill <<<<<<");
        MyAddressAllocator::activation_malloc();
        uint32_t Demb = Config::system_config.model_n_embd;        // generate activation for multi batches
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }

        auto ffn_inputs = test_attn_block(my_inputs, 0, _breq->_reqs);
        auto my_outputs = test_ffn_block(ffn_inputs, 0);
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "llama_ffn") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: llama_ffn <<<<<<");
        MyAddressAllocator::activation_malloc();
        uint32_t Demb = Config::system_config.model_n_embd;
        // generate activation for multi batches
        std::vector<Ptr<MyTensor>> my_inputs;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> input_dim = {req->input_size, Demb} ;
            auto my_input = std::make_shared<MyTensor>("input", input_dim, TensorType::ACT, true);
            my_inputs.push_back(my_input);
        }
        auto my_outputs = test_llama_ffn_block(my_inputs, 0);  // layer 0
        find_executable_node(my_inputs);
    }
    else if (_test_multi_layer_name == "data_convert_weights") {
        spdlog::info(">>>>>> Initialize Multi Layer Model Program: data_convert_weights <<<<<<");
        AllocationScheme src_scheme = MyAddressAllocator::allocation_scheme;
        AllocationScheme dst_scheme;
        if (src_scheme == AllocationScheme::NPU) {
            dst_scheme = AllocationScheme::IANUS;
        }
        else if (src_scheme == AllocationScheme::IANUS) {
            dst_scheme = AllocationScheme::NPU;
        }
        else {
            throw std::runtime_error("data_convert_weights only supports NPU <-> IANUS");
        }

        std::vector<Ptr<MyTensor>> source_weights;
        auto add_weight_convert = [&](int layer, const std::string& block_type, const std::string& operation_type) {
            auto params = _model->get_my_params(layer, block_type, operation_type);
            auto weight = params[0];
            source_weights.push_back(weight);
            auto convert = add_op(std::make_shared<DataConvert>(
                name_gen(weight->get_name(), "DataConvert"), src_scheme, dst_scheme));
            convert->get_my_outputs({weight}, TensorType::WGT);
        };

        for (int layer = 0; layer < Config::system_config.model_n_layer; ++layer) {
            add_weight_convert(layer, BlockType::Attention, OperationType::QGen);
            add_weight_convert(layer, BlockType::Attention, OperationType::KGen);
            add_weight_convert(layer, BlockType::Attention, OperationType::VGen);
            add_weight_convert(layer, BlockType::Attention, OperationType::Projection);
            if (is_llama_model_name(Config::system_config.model_name)) {
                add_weight_convert(layer, BlockType::FeedForward, OperationType::GateProj);
                add_weight_convert(layer, BlockType::FeedForward, OperationType::UpProj);
                add_weight_convert(layer, BlockType::FeedForward, OperationType::DownProj);
            }
            else {
                add_weight_convert(layer, BlockType::FeedForward, OperationType::FullyConnected1);
                add_weight_convert(layer, BlockType::FeedForward, OperationType::FullyConnected2);
            }
        }

        find_executable_node(source_weights);
    }
    else if (_test_multi_layer_name == "llama_decode_ffn") {

    }
    else if (_test_multi_layer_name == "llama_attn") {

    }
    else if (_test_multi_layer_name == "llama_decode_attn") {

    }
    else {
        throw std::runtime_error("Invalid multi layer name");
    }

}


std::vector<Ptr<MyTensor>> StageProgram::init_program_att_block(std::vector<Ptr<MyTensor>> inputs, int layer, std::vector<std::shared_ptr<InferRequest>> reqs) {
    if (_stage == Stage::Prefill) {
        // Prefill Stage
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::Attention);

        // Norm
        auto norm = create_norm_op(prefix, layer, BlockType::Attention);
        inputs = get_my_outputs(norm, inputs);

        //QKV Gen
        auto QGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QGen),
            _model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
        QGen->set_apply_rope();
        auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

        auto KGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::KGen),
            _model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
        KGen->set_apply_rope();
        auto K = KGen->get_my_outputs(inputs, TensorType::KCache);

        auto VGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::VGen),
            _model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));
        auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

        for (uint32_t i = 0; i < reqs.size(); ++i) {
            auto req = reqs[i];
            _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::KGen)] = K[i];
            _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::VGen)] = V[i];
        }

        //QKT & SV
        auto QK = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QKGEMM), K));
        auto qk = QK->get_my_outputs(Q, TensorType::ACT);
        auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
        auto s = get_my_outputs(S, qk);
        auto SV = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::SVGEMM), V));
        auto sv = SV->get_my_outputs(s, TensorType::ACT);

        //Projection
        auto projection = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::Projection),
            _model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
        inputs = projection->get_my_outputs(sv, TensorType::ACT);

        //Residual
        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);

        return inputs;
    }
    else if (_stage == Stage::Decode or _stage == Stage::NPU_Decode) {
        // Decode Stage
        if (_stage_platform == StagePlatform::SA) {  // GEMV
            auto res_buf = inputs;
            std::string prefix = name_gen(LAYER(layer), BlockType::Attention);

            std::vector<Ptr<MyTensor>> stored_KCaches;
            std::vector<Ptr<MyTensor>> stored_VCaches;
            for (uint32_t i = 0; i < reqs.size(); ++i) {
                auto req = reqs[i];
                auto stored_KCache = _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::KGen)];
                auto stored_VCache = _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::VGen)];
                stored_KCache->clear_child_nodes();
                stored_VCache->clear_child_nodes();
                stored_KCaches.push_back(stored_KCache);
                stored_VCaches.push_back(stored_VCache);
            }

            auto norm = create_norm_op(prefix, layer, BlockType::Attention);
            inputs = get_my_outputs(norm, inputs);

            //QKV Gen by PIMGEMV
            auto QGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::QGen),_model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
            QGen->set_apply_rope(is_llama_model_name(Config::system_config.model_name));
            auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

            auto KGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::KGen),_model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
            KGen->set_apply_rope(is_llama_model_name(Config::system_config.model_name));
            auto VGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::VGen),_model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));

            auto KCache = KGen->kvcache_append(inputs, stored_KCaches, TensorType::KCache);  //auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
            auto VCache = VGen->kvcache_append(inputs, stored_VCaches, TensorType::VCache);  //auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

            //QKT & SV by PIMGEMV
            auto QK = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::QKGEMV), KCache));
            auto qk = QK->get_my_outputs(Q, TensorType::ACT);
            auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
            auto s = get_my_outputs(S, qk);
            auto SV = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::SVGEMV), VCache));
            auto sv = SV->get_my_outputs(s, TensorType::ACT);

            //Projection by PIMGEMV
            auto projection = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::Projection),_model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
            inputs = projection->get_my_outputs(sv, TensorType::ACT);

            //Residual
            auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
            inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
            inputs = get_my_outputs(residual, inputs);

            return inputs;
        }
        else if (_stage_platform == StagePlatform::PIM) { // PIM-GEMV
            auto res_buf = inputs;
            std::string prefix = name_gen(LAYER(layer), BlockType::Attention);

            std::vector<Ptr<MyTensor>> stored_KCaches;
            std::vector<Ptr<MyTensor>> stored_VCaches;
            for (uint32_t i = 0; i < reqs.size(); ++i) {
                auto req = reqs[i];
                auto stored_KCache = _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::KGen)];
                auto stored_VCache = _model->_stored_KVCache[req.get()][name_gen(prefix, OperationType::VGen)];
                stored_KCache->clear_child_nodes();
                stored_VCache->clear_child_nodes();
                stored_KCaches.push_back(stored_KCache);
                stored_VCaches.push_back(stored_VCache);
            }

            auto norm = create_norm_op(prefix, layer, BlockType::Attention);
            inputs = get_my_outputs(norm, inputs);

            //QKV Gen by PIMGEMV
            auto QGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QGen),_model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
            QGen->set_apply_rope();
            auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

            auto KGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::KGen),_model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
            KGen->set_apply_rope();
            auto VGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::VGen),_model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));

            auto KCache = KGen->kvcache_append(inputs, stored_KCaches, TensorType::KCache);  //auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
            auto VCache = VGen->kvcache_append(inputs, stored_VCaches, TensorType::VCache);  //auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

            //QKT & SV by PIMGEMV
            auto QK = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QKPIMGEMV), KCache));
            auto qk = QK->get_my_outputs(Q, TensorType::ACT);
            auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
            auto s = get_my_outputs(S, qk);
            auto SV = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::SVPIMGEMV), VCache));
            auto sv = SV->get_my_outputs(s, TensorType::ACT);

            //Projection by PIMGEMV
            auto projection = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::Projection),_model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
            inputs = projection->get_my_outputs(sv, TensorType::ACT);

            //Residual
            auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
            inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
            inputs = get_my_outputs(residual, inputs);

            return inputs;
        }
        else {
            throw std::runtime_error("The Supported StagePlatform Only SA and PIM");
        }
    }
    else {
        throw std::runtime_error("Not Support Inference Stage");
    }
}


std::vector<Ptr<MyTensor>> StageProgram::init_program_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer) {
    std::string model_name = Config::system_config.model_name;
    
    if (_stage == Stage::Prefill) {
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
        // create operations
        auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
        inputs = get_my_outputs(norm, inputs);

        auto fc1 = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::FullyConnected1),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
        inputs = get_my_outputs(fc1, inputs);

        // LLaMA2 uses SwiGLU instead of GELU, OPT and GPT-3 use GELU (or ReLU, but we use GELU here)
        if (model_name.find("LLAMA2") != std::string::npos || model_name.find("Llama-2") != std::string::npos) {
            // Note: properly we should have gate_proj and up_proj and multiply them.
            // For now, we approximate SwiGLU with a Gelu operation node if we only simulate one FC1
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        } else {
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        }

        auto fc2 = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::FullyConnected2),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
        inputs = get_my_outputs(fc2, inputs);

        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);
        return inputs;
    }
    else if (_stage == Stage::Decode or _stage == Stage::NPU_Decode) {
        if (_stage_platform == StagePlatform::SA) {
            auto res_buf = inputs;
            std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
            // create operations
            auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
            inputs = get_my_outputs(norm, inputs);

            auto fc1 = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::FullyConnected1),
                _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
            inputs = get_my_outputs(fc1, inputs);

            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);

            auto fc2 = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::FullyConnected2),
                _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
            inputs = get_my_outputs(fc2, inputs);

            auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
            inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
            inputs = get_my_outputs(residual, inputs);
            return inputs;
        }
        else if (_stage_platform == StagePlatform::PIM) {
            auto res_buf = inputs;
            std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
            // create operations
            auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
            inputs = get_my_outputs(norm, inputs);

            auto fc1 = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::FullyConnected1),
                _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
            inputs = get_my_outputs(fc1, inputs);

            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);

            auto fc2 = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::FullyConnected2),
                _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
            inputs = get_my_outputs(fc2, inputs);

            auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
            inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
            inputs = get_my_outputs(residual, inputs);
            return inputs;
        }
        else {
            throw std::runtime_error("The Supported StagePlatform Only SA and PIM");
        }
    }
    else {
        throw std::runtime_error("Not Support Inference Stage");
    }
}

std::vector<Ptr<MyTensor>> StageProgram::test_llama_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer) {
    auto res_buf = inputs;
    std::string prefix = name_gen(stageToString(_stage), LAYER(layer), BlockType::FeedForward);

    auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
    auto norm_outputs = get_my_outputs(norm, inputs);

    auto gate_proj = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::GateProj),
        _model->get_my_params(layer, BlockType::FeedForward, OperationType::GateProj)));
    auto gate = gate_proj->get_my_outputs(norm_outputs, TensorType::ACT);

    auto silu = add_op(std::make_shared<SiLU>(name_gen(prefix, OperationType::SiLU)));
    auto silu_gate = silu->get_my_outputs(gate, TensorType::ACT);

    auto up_proj = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::UpProj),
        _model->get_my_params(layer, BlockType::FeedForward, OperationType::UpProj)));
    auto up = up_proj->get_my_outputs(norm_outputs, TensorType::ACT);

    std::vector<Ptr<MyTensor>> mul_inputs;
    mul_inputs.reserve(silu_gate.size() + up.size());  // mul inputs, half the silu gate, half the up_proj
    mul_inputs.insert(mul_inputs.end(), silu_gate.begin(), silu_gate.end());
    mul_inputs.insert(mul_inputs.end(), up.begin(), up.end());

    auto mul = add_op(std::make_shared<Mul>(name_gen(prefix, OperationType::Mul)));
    auto gated_hidden = mul->get_my_outputs(mul_inputs, TensorType::ACT);

    auto down_proj = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::DownProj),
        _model->get_my_params(layer, BlockType::FeedForward, OperationType::DownProj)));
    auto outputs = down_proj->get_my_outputs(gated_hidden, TensorType::ACT);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    outputs.insert(outputs.end(), res_buf.begin(), res_buf.end());
    return residual->get_my_outputs(outputs, TensorType::ACT);
}



std::vector<Ptr<MyTensor>> StageProgram::test_ffn_block(std::vector<Ptr<MyTensor>> inputs, int layer) {
    auto res_buf = inputs;
    std::string prefix = name_gen(stageToString(_stage), LAYER(layer), BlockType::FeedForward);
    std::string model_name = Config::system_config.model_name;

    // create operations
    auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
    inputs = get_my_outputs(norm, inputs);

    auto fc1 = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::FullyConnected1),
        _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
    inputs = get_my_outputs(fc1, inputs);

    auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
    inputs = get_my_outputs(gelu, inputs);

    auto fc2 = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::FullyConnected2),
        _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
    inputs = get_my_outputs(fc2, inputs);

    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));

    inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
    inputs = get_my_outputs(residual, inputs);
    return inputs;
}


std::vector<Ptr<MyTensor>> StageProgram::test_attn_block(std::vector<Ptr<MyTensor>> inputs, int layer, InferRequest* req) {
    // 多Batch不这么用了
    auto res_buf = inputs;
    std::string prefix = name_gen(stageToString(_stage), LAYER(layer), BlockType::Attention);

    // Layernorm
    auto norm = create_norm_op(prefix, layer, BlockType::Attention);
    inputs = get_my_outputs(norm, inputs);

    // QKV Gen
    auto QGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QGen),
        _model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
    auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);
    auto KGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::KGen),
        _model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
    auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
    auto VGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::VGen),
        _model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));
    auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

    auto stored_KCache = _model->_stored_KVCache[req][name_gen(prefix, OperationType::KGen)];
    auto stored_VCache = _model->_stored_KVCache[req][name_gen(prefix, OperationType::VGen)];

    //QKT & SV
    auto QK = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QKGEMM), K));
    auto qk = QK->get_my_outputs(std::vector{Q[0], K[0]}, TensorType::ACT);
    auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
    auto s = get_my_outputs(S, qk);
    auto SV = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::SVGEMM), V));
    auto sv = SV->get_my_outputs(std::vector{s[0], V[0]}, TensorType::ACT);

    //Projection
    auto projection = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::Projection),
        _model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = projection->get_my_outputs(sv, TensorType::ACT);

    //Residual
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
    inputs = get_my_outputs(residual, inputs);

    return inputs;
}


std::vector<Ptr<MyTensor>> StageProgram::test_attn_block(std::vector<Ptr<MyTensor>> inputs, int layer, std::vector<std::shared_ptr<InferRequest>> reqs) {
    auto res_buf = inputs;
    std::string prefix = name_gen(stageToString(_stage), LAYER(layer), BlockType::Attention);
    // LayerNorm
    auto norm = create_norm_op(prefix, layer, BlockType::Attention);
    inputs = get_my_outputs(norm, inputs);

    auto QGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QGen),
    _model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
    QGen->set_apply_rope();
    auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

    auto KGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::KGen),
        _model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
    KGen->set_apply_rope();
    auto K = KGen->get_my_outputs(inputs, TensorType::KCache);

    auto VGen = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::VGen),
        _model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));
    auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

    // QKT & SV
    auto QK = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::QKGEMM), K));
    auto qk = QK->get_my_outputs(Q, TensorType::ACT);
    auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
    auto s = get_my_outputs(S, qk);
    auto SV = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::SVGEMM), V));
    auto sv = SV->get_my_outputs(s, TensorType::ACT);

    //Projection
    auto projection = add_op(std::make_shared<GEMM>(name_gen(prefix, OperationType::Projection),
        _model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = projection->get_my_outputs(sv, TensorType::ACT);

    //Residual
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
    inputs = get_my_outputs(residual, inputs);
    return inputs;
}



std::vector<Ptr<MyTensor>> StageProgram::test_attention_block(std::vector<Ptr<MyTensor>> inputs) {
    int layer = 0;
    auto res_buf = inputs;
    std::string prefix = name_gen(stageToString(_stage), LAYER(layer), BlockType::Attention);

    auto norm = create_norm_op(prefix, layer, BlockType::Attention);
    inputs = get_my_outputs(norm, inputs);

    //QKV Gen by PIMGEMV
    auto QGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QGen),_model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
    auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

    auto KGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::KGen),_model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
    auto VGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::VGen),_model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));

    // The KCache and VCache initialized for test, replace by recorded kcache in prefill stage
    std::vector<uint32_t> cache_dim{300, model_kv_cache_width()};
    auto my_kcache = std::make_shared<MyTensor>("Example_KCache", cache_dim, TensorType::KCache, true);
    auto my_vcache = std::make_shared<MyTensor>("Example_VCache", cache_dim, TensorType::VCache, true);

    auto KCache = KGen->kvcache_append(inputs, {my_kcache}, TensorType::KCache);  //auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
    auto VCache = VGen->kvcache_append(inputs, {my_vcache}, TensorType::VCache);  //auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

    // QKT & SV by PIMGEMV
    auto QK = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::PIMGEMV), KCache));
    auto qk = QK->get_my_outputs(std::vector{Q[0], KCache[0]}, TensorType::ACT);
    auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
    auto s = get_my_outputs(S, qk);
    auto SV = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QKGEMM), VCache));
    auto sv = SV->get_my_outputs(std::vector{s[0], VCache[0]}, TensorType::ACT);

    // Projection by PIMGEMV
    auto projection = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::Projection),_model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
    inputs = projection->get_my_outputs(sv, TensorType::ACT);

    // Residual
    auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
    inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
    inputs = get_my_outputs(residual, inputs);

    return inputs;
}

std::vector<Ptr<MyTensor>> StageProgram::test_decode_stage_atten(std::vector<Ptr<MyTensor>> inputs, int layer) {
    uint32_t batch_size = inputs.size();

    if (_stage_platform == StagePlatform::SA) {  // GEMV
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::Attention);

        std::vector<Ptr<MyTensor>> stored_KCaches;
        std::vector<Ptr<MyTensor>> stored_VCaches;
        // The KCache and VCache initialized for test, replace by recorded kcache in prefill stage
        for (uint32_t i = 0; i < batch_size; i++) {  // Initialize the KV cache used by this test.
            std::vector<uint32_t> cache_dim{_breq->_reqs[i]->input_size, model_kv_cache_width()};
            auto stored_KCache = std::make_shared<MyTensor>("Example_KCache", cache_dim, TensorType::KCache, true);
            auto stored_VCache = std::make_shared<MyTensor>("Example_VCache", cache_dim, TensorType::VCache, true);
            stored_KCaches.push_back(stored_KCache);
            stored_VCaches.push_back(stored_VCache);
        }

        for (uint32_t i = 0; i < batch_size; i++) {
            stored_KCaches[i]->clear_child_nodes();
            stored_VCaches[i]->clear_child_nodes();
        }
        /*
        std::vector<Ptr<MyTensor>> my_caches;
        for (auto req : _breq->_reqs) {
            std::vector<uint32_t> cache_dim = {req->input_size, Demb};
            auto my_cache = std::make_shared<MyTensor>("PIMGEMV_Cache", cache_dim, TensorType::KCache, true);
            my_caches.push_back(my_cache);
        }
        auto my_outputs = my_single_op->kvcache_append(my_inputs, my_caches, TensorType::VCache);

        auto stored_KCache = std::make_shared<MyTensor>("Example_KCache", cache_dim, TensorType::KCache, true);
        auto stored_VCache = std::make_shared<MyTensor>("Example_VCache", cache_dim, TensorType::VCache, true);

        stored_KCache->clear_child_nodes();
        stored_VCache->clear_child_nodes();
        */
        auto norm = create_norm_op(prefix, layer, BlockType::Attention);
        inputs = get_my_outputs(norm, inputs);

        // QKV Gen by PIMGEMV
        auto QGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::QGen),_model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
        auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

        auto KGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::KGen),_model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
        auto VGen = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::VGen),_model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));

        auto KCache = KGen->kvcache_append(inputs, stored_KCaches, TensorType::KCache);  //auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
        auto VCache = VGen->kvcache_append(inputs, stored_VCaches, TensorType::VCache);  //auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

        // QKT & SV by PIMGEMV
        auto QK = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::QKGEMV), KCache));
        auto qk = QK->get_my_outputs(Q, TensorType::ACT);
        auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
        auto s = get_my_outputs(S, qk);
        auto SV = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::SVGEMV), VCache));
        auto sv = SV->get_my_outputs(s, TensorType::ACT);

        // Projection by PIMGEMV
        auto projection = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::Projection),_model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
        inputs = projection->get_my_outputs(sv, TensorType::ACT);

        // Residual
        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);

        return inputs;
    }
    else if (_stage_platform == StagePlatform::PIM) { // PIM-GEMV
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::Attention);
        // The KCache and VCache initialized for test, replace by recorded kcache in prefill stage
        std::vector<Ptr<MyTensor>> stored_KCaches;
        std::vector<Ptr<MyTensor>> stored_VCaches;
        // The KCache and VCache initialized for test, replace by recorded kcache in prefill stage
        for (uint32_t i = 0; i < batch_size; i++) {  // Initialize the KV cache used by this test.
            std::vector<uint32_t> cache_dim{_breq->_reqs[i]->input_size, model_kv_cache_width()};
            auto stored_KCache = std::make_shared<MyTensor>("Example_KCache", cache_dim, TensorType::KCache, true);
            auto stored_VCache = std::make_shared<MyTensor>("Example_VCache", cache_dim, TensorType::VCache, true);
            stored_KCaches.push_back(stored_KCache);
            stored_VCaches.push_back(stored_VCache);
        }

        for (uint32_t i = 0; i < batch_size; i++) {
            stored_KCaches[i]->clear_child_nodes();
            stored_VCaches[i]->clear_child_nodes();
        }

        /*
        std::vector<uint32_t> cache_dim{256, model_kv_cache_width()};
        auto stored_KCache = std::make_shared<MyTensor>("Example_KCache", cache_dim, TensorType::KCache, true);
        auto stored_VCache = std::make_shared<MyTensor>("Example_VCache", cache_dim, TensorType::VCache, true);
        stored_KCache->clear_child_nodes();
        stored_VCache->clear_child_nodes();
        */
        auto norm = create_norm_op(prefix, layer, BlockType::Attention);
        inputs = get_my_outputs(norm, inputs);

        // QKV Gen by PIMGEMV
        auto QGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QGen),_model->get_my_params(layer, BlockType::Attention, OperationType::QGen)));
        auto Q = QGen->get_my_outputs(inputs, TensorType::ACT);

        auto KGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::KGen),_model->get_my_params(layer, BlockType::Attention, OperationType::KGen)));
        auto VGen = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::VGen),_model->get_my_params(layer, BlockType::Attention, OperationType::VGen)));

        auto KCache = KGen->kvcache_append(inputs, stored_KCaches, TensorType::KCache);  //auto K = KGen->get_my_outputs(inputs, TensorType::KCache);
        auto VCache = VGen->kvcache_append(inputs, stored_VCaches, TensorType::VCache);  //auto V = VGen->get_my_outputs(inputs, TensorType::VCache);

        //QKT & SV by PIMGEMV
        auto QK = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::QKPIMGEMV), KCache));
        auto qk = QK->get_my_outputs(Q, TensorType::ACT);
        auto S = add_op(std::make_shared<Softmax>(name_gen(prefix, OperationType::SoftMax)));
        auto s = get_my_outputs(S, qk);
        auto SV = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::SVPIMGEMV), VCache));
        auto sv = SV->get_my_outputs(s, TensorType::ACT);

        //Projection by PIMGEMV
        auto projection = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::Projection),_model->get_my_params(layer, BlockType::Attention, OperationType::Projection)));
        inputs = projection->get_my_outputs(sv, TensorType::ACT);

        //Residual
        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));
        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);

        return inputs;
    }
    else {
        throw std::runtime_error("The Supported StagePlatform Only SA and PIM");
    }
}


std::vector<Ptr<MyTensor>> StageProgram::test_decode_stage_ffn(std::vector<Ptr<MyTensor>> inputs, int layer) {
    std::string model_name = Config::system_config.model_name;

    if (_stage_platform == StagePlatform::SA) {
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
        // create operations
        auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
        inputs = get_my_outputs(norm, inputs);

        auto fc1 = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::FullyConnected1),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
        inputs = get_my_outputs(fc1, inputs);

        if (model_name.find("LLAMA2") != std::string::npos || model_name.find("Llama-2") != std::string::npos) {
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        } else {
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        }

        auto fc2 = add_op(std::make_shared<GEMV>(name_gen(prefix, OperationType::FullyConnected2),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
        inputs = get_my_outputs(fc2, inputs);

        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));

        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);
        return inputs;
    }
    else if (_stage_platform == StagePlatform::PIM) {
        auto res_buf = inputs;
        std::string prefix = name_gen(LAYER(layer), BlockType::FeedForward);
        // create operations
        auto norm = create_norm_op(prefix, layer, BlockType::FeedForward);
        inputs = get_my_outputs(norm, inputs);

        auto fc1 = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::FullyConnected1),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected1)));
        inputs = get_my_outputs(fc1, inputs);

        if (model_name.find("LLAMA2") != std::string::npos || model_name.find("Llama-2") != std::string::npos) {
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        } else {
            auto gelu = add_op(std::make_shared<Gelu>(name_gen(prefix, OperationType::Gelu)));
            inputs = get_my_outputs(gelu, inputs);
        }

        auto fc2 = add_op(std::make_shared<PIMGEMV>(name_gen(prefix, OperationType::FullyConnected2),
            _model->get_my_params(layer, BlockType::FeedForward, OperationType::FullyConnected2)));
        inputs = get_my_outputs(fc2, inputs);

        auto residual = add_op(std::make_shared<Add>(name_gen(prefix, OperationType::Residual)));

        inputs.insert(inputs.end(), res_buf.begin(), res_buf.end());
        inputs = get_my_outputs(residual, inputs);
        return inputs;
    }
    else {
        throw std::runtime_error("The Supported StagePlatform Only SA and PIM");
    }
}
Ptr<Operation> StageProgram::add_op(std::shared_ptr<Operation> op) {
    // spdlog::info("operation {} added. add_op", op->get_name());
    _op_map[op->get_id()] = op;
    return op;
}

std::vector<Ptr<BTensor>> StageProgram::get_outputs(Ptr<Operation> op,
                                                    std::vector<Ptr<BTensor>> inputs) {
    return op->get_outputs(inputs);
}

std::vector<Ptr<MyTensor>> StageProgram::get_my_outputs(Ptr<Operation> op, std::vector<Ptr<MyTensor>> inputs) {
    return op->get_my_outputs(inputs);
}

void StageProgram::find_executable_node(Ptr<BTensor> tensor) {
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            _executable_operations.push_back(op);
        }
    }
}


void StageProgram::find_executable_node(Ptr<MyTensor> tensor) {
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            _executable_operations.push_back(op);
        }
    }
}


void StageProgram::find_executable_node(std::vector<Ptr<MyTensor>> tensors) {
    for (const auto& tensor : tensors) {
        for (auto op : tensor->get_child_nodes()) {
            // spdlog::info("initializing operation {} ...", op->get_name());
            if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
                _executable_operations.push_back(op);
            }
        }
    }
}


bool StageProgram::check_exist_in_executable(uint32_t op_id) {
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (op_id == (*iter)->get_id()) {
            return true;
        }
    }
    return false;
}

void StageProgram::finish_operation(uint32_t id) {
    _op_map[id]->set_finish();
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        // spdlog::info("iterating operation: {}", (*iter)->get_name());
        if (id == (*iter)->get_id()) {
            // spdlog::info("erasing operation: {}", (*iter)->get_name());
            _executable_operations.erase(iter);
            break;
        }
    }

    for (auto op : _op_map[id]->get_child_nodes()) {
        // spdlog::info("finding operation: {} / {} ", op->get_name(), op->get_id());
        if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
            // spdlog::info("found operation: {}", op->get_name());
            _executable_operations.push_back(op);
        }
    }
}

bool StageProgram::check_finish() {
    bool finish = true;
    for (auto const &[key, val] : _op_map) {
        finish = finish && val->check_finish();
    }

    return finish;
}

std::vector<OperationStat> StageProgram::list_operation_stat() {
    std::vector<OperationStat> ret;
    for (auto &[key, val] : _op_map) {
        ret.push_back(val->get_stat());
    }

    return ret;
}

void StageProgram::finish_operation_tile(Tile &tile) {
    _op_map[tile.operation_id]->reduce_tile(tile);
}

/**
 * logger function for StageProgram
 * TODO: log file name is tentative. think of fname rule
 */
void StageProgram::log() {
    std::string fname = Config::system_config.log_dir + "/" + _name;
    Logger::log(list_operation_stat(), fname);
}
