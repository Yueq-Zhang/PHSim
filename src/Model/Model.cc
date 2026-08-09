#include "Model.h"

#include <utility>

namespace BlockType {
std::string Attention = "attn";
std::string FeedForward = "ffn";
}  // namespace BlockType

namespace OperationType {
std::string LayerNorm = "layernorm";
std::string RMSNorm = "rmsnorm";
std::string QKVGen = "QKVgen";
std::string QGen = "QGen";
std::string KGen = "KGen";
std::string VGen = "VGen";
std::string Projection = "proj";
std::string FullyConnected1 = "fc1";
std::string FullyConnected2 = "fc2";
std::string GateProj = "gate_proj";
std::string UpProj = "up_proj";
std::string DownProj = "down_proj";
std::string LmHead = "lmhead";

std::string QKVSplit = "QKVsplit";
std::string QKGEMM = "QKGEMM";
std::string SoftMax = "softmax";
std::string SVGEMM = "SVGEMM";
std::string LsVGEMM = "LsVmm";
std::string QKGEMV = "QKGEMV";
std::string SVGEMV = "SVGEMV";
std::string QKPIMGEMV = "QKPIMGEMV";
std::string SVPIMGEMV = "SVPIMGEMV";

std::string AReshape = "Areshape";
std::string Residual = "res";
std::string Gelu = "gelu";
std::string SiLU = "silu";
std::string Mul = "mul";
std::string BatchSplit = "BSplit";
std::string BatchConcat = "BConcat";
std::string KCacheConcat = "Kccat";
std::string VCacheConcat = "Vccat";
std::string VConcat = "Vcat";
std::string PIMGEMVSoftmax = "PIMGEMVSoftmax";
std::string PIMGEMVAdd = "PIMGEMVAdd";
std::string NeuPIMSLogitSoftmax = "NeuPIMSLogitSoftmax";
std::string Attention = "Attention";
std::string Microbench = "Microbench";
std::string NeuPIMSAttend = "NeuPIMSAttend";
std::string FusedMHA = "FusedMHA";
std::string PIMGEMV = "PIMGEMV";
}  // namespace OperationType

namespace ParameterType {
std::string Weight = "weight";
std::string Bias = "bias";
}  // namespace ParameterType

Model::Model(const SysConfig& config, std::string name) : _config(config) {
    _name = std::move(name);
    _root_node_id = generate_id();
    init_my_params();
}

Model::~Model() {
    _operation_map.clear();
    _tensor_map.clear();
    _my_wgt_map.clear();
    _executable_operations.clear();
    _stored_KVCache.clear();
    _input_tensor.reset();
}


void Model::init_my_params() {
    if (Config::system_config.test_single_op) {
        spdlog::info("Single Op Test mode");
    }
    else{
        std::string model_name = Config::system_config.model_name;
        uint32_t hidden_size = Config::system_config.model_n_embd;
        uint32_t tp_size = Config::system_config.n_tp;
        uint32_t head_dim = hidden_size / Config::system_config.model_n_head;
        bool is_llama_model = is_llama_model_name(model_name);
        bool use_gqa = is_llama_model && Config::system_config.model_n_kv_head != Config::system_config.model_n_head;
        uint32_t kv_hidden_size = use_gqa ? Config::system_config.model_n_kv_head * head_dim : hidden_size;
        if (use_gqa && kv_hidden_size % tp_size != 0) {
            throw std::runtime_error(fmt::format("Invalid GQA kv_hidden_size {} for tp_size {}",
                kv_hidden_size, tp_size));
        }
        std::string norm_type = is_llama_model ? OperationType::RMSNorm : OperationType::LayerNorm;
        
        for (int i = 0; i < _config.model_n_layer; ++i) {
            auto attn = name_gen(LAYER(i), BlockType::Attention);

            create_my_weight(name_gen(attn, norm_type, ParameterType::Weight), {hidden_size});
            if (!is_llama_model) {
                create_my_weight(name_gen(attn, norm_type, ParameterType::Bias), {hidden_size});
            }
            
            // QKV projections
            create_my_weight(name_gen(attn, OperationType::QGen, ParameterType::Weight), {hidden_size, hidden_size / tp_size});
            create_my_weight(name_gen(attn, OperationType::QGen, ParameterType::Bias), {hidden_size / tp_size});
            create_my_weight(name_gen(attn, OperationType::KGen, ParameterType::Weight), {hidden_size, kv_hidden_size / tp_size});
            create_my_weight(name_gen(attn, OperationType::KGen, ParameterType::Bias), {kv_hidden_size / tp_size});
            create_my_weight(name_gen(attn, OperationType::VGen, ParameterType::Weight), {hidden_size, kv_hidden_size / tp_size});
            create_my_weight(name_gen(attn, OperationType::VGen, ParameterType::Bias), {kv_hidden_size / tp_size});
            
            // Out projection
            create_my_weight(name_gen(attn, OperationType::Projection, ParameterType::Weight), {hidden_size / tp_size, hidden_size});
            create_my_weight(name_gen(attn, OperationType::Projection, ParameterType::Bias), {hidden_size});

            auto ffn = name_gen(LAYER(i), BlockType::FeedForward);
            create_my_weight(name_gen(ffn, norm_type, ParameterType::Weight), {hidden_size});
            if (!is_llama_model) {
                create_my_weight(name_gen(ffn, norm_type, ParameterType::Bias), {hidden_size});
            }

            // FFN configuration based on model type
            uint32_t intermediate_size;
            if (is_llama_model) {
                // LLaMA2 uses SwiGLU, so it has up_proj, gate_proj, and down_proj
                // Typically intermediate_size is around 4/3 * hidden_size * 2 (with specific multiples)
                // We'll use 4 * hidden_size as a default if not specified elsewhere, but for llama we simulate 3 matrices
                intermediate_size = static_cast<uint32_t>(3.5 * hidden_size); // Or read from config if available
                create_my_weight(name_gen(ffn, OperationType::GateProj, ParameterType::Weight), {hidden_size, intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::GateProj, ParameterType::Bias), {intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::UpProj, ParameterType::Weight), {hidden_size, intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::UpProj, ParameterType::Bias), {intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::DownProj, ParameterType::Weight), {intermediate_size / tp_size, hidden_size});
                create_my_weight(name_gen(ffn, OperationType::DownProj, ParameterType::Bias), {hidden_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight), {hidden_size, intermediate_size / tp_size}); // up_proj
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias), {intermediate_size / tp_size});
                // create_my_weight(name_gen(ffn, "gate_proj", ParameterType::Weight), {hidden_size, intermediate_size / tp_size}); // gate_proj (omitted for now to match interface or we add it)
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight), {intermediate_size / tp_size, hidden_size}); // down_proj
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias), {hidden_size});
            } else if (model_name.find("OPT") != std::string::npos) {
                intermediate_size = 4 * hidden_size;
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight), {hidden_size, intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias), {intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight), {intermediate_size / tp_size, hidden_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias), {hidden_size});
            } else {
                // Default GPT-3 style
                intermediate_size = 4 * hidden_size;
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight), {hidden_size, intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias), {intermediate_size / tp_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight), {intermediate_size / tp_size, hidden_size});
                create_my_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias), {hidden_size});
            }
        }
        // LM head: both encoder, decoder are GEMV
        // create_my_weight(name_gen(OperationType::LmHead, ParameterType::Weight),{Config::system_config.model_n_embd, Config::system_config.model_vocab_size});

        // In advance, weight size Calculation
        _wgt_size = 0;
        for (auto const &[tensor_name, tensor] : _my_wgt_map) {
            auto dims = tensor->_dims;
            _wgt_size = MyAddressAllocator::precision_weight;
            for (auto dim : dims) {
                _wgt_size *= dim;
            }
        }
        spdlog::info("Total weight size: {}", _wgt_size);
    }
}

Ptr<MyTensor> Model::find_my_tensor(std::string name) {return _my_wgt_map[name];}


std::vector<Ptr<MyTensor>> Model::get_my_params(int layer_idx, std::string block_type, std::string operation_type) {
    std::string prefix = name_gen(LAYER(layer_idx), block_type, operation_type);
    Ptr<MyTensor> wgt = find_my_tensor(name_gen(prefix, ParameterType::Weight));
    if (operation_type == OperationType::RMSNorm) {
        return {wgt};
    }
    Ptr<MyTensor> bias = find_my_tensor(name_gen(prefix, ParameterType::Bias));
    return {wgt, bias};
}

void Model::find_executable_node(std::shared_ptr<MyTensor> tensor) {
    // spdlog::info("intializing from tensor {}", tensor_id);
    for (auto op : tensor->get_child_nodes()) {
        // spdlog::info("initializing operation {} ...", op->get_name());
        if (op->check_executable()) {
            // spdlog::info("success initializing operation {}!",
            // op->get_name());
            _executable_operations.push_back(op);
        }
    }
}

Model::Model(const Model &model) : _config(model._config) {
    _name = model._name;
    _root_node_id = model._root_node_id;
    _input_tensor = model._input_tensor;

    _input_name = model._input_name;
    _input_dim = model._input_dim;
    _wgt_size = model._wgt_size;

    for (auto const &[key, val] : model._tensor_map) {
        _tensor_map[key] = val;
    }
    for (auto const &[key, val] : model._operation_map) {
        _operation_map[key] = val;
    }
    for (auto operation : model._executable_operations) {
        _executable_operations.push_back(_operation_map[operation->get_id()]);
        spdlog::trace("add op {0:x}", fmt::ptr(_executable_operations.front()));
    }
}

std::shared_ptr<MyTensor> Model::create_my_weight(std::string name, std::vector<uint32_t> dims) {
    auto tensor = std::make_shared<MyTensor>(name, dims, TensorType::WGT, true);
    _my_wgt_map[name] = tensor;
    return tensor;
}

// auto tensor = std::make_shared<BatchedTensor>(name, dims, TensorBufType::WGT, true);

// input: operation id
// erase target operation and insert readied operation.
void Model::finish_operation(uint32_t id) {
    _operation_map[id]->set_finish();
    for (auto iter = _executable_operations.begin(); iter != _executable_operations.end(); iter++) {
        if (id == (*iter)->get_id()) {
            _executable_operations.erase(iter);
            break;
        }
    }
    // log_operations();
    for (auto op : _operation_map[id]->get_child_nodes()) {
        if (op->check_executable() && !check_exist_in_executable(op->get_id())) {
            _executable_operations.push_back(op);
        }
    }
}

void Model::finish_operation_tile(uint32_t id, Tile &tile) {
    _operation_map[id]->reduce_tile(tile);
}

std::vector<std::shared_ptr<Operation>> Model::get_executable_operations() {
    return _executable_operations;
}

bool Model::check_exist_in_executable(uint32_t op_id) {
    for (auto & _executable_operation : _executable_operations) {
        if (op_id == _executable_operation->get_id()) {
            return true;
        }
    }
    return false;
}

uint64_t Model::get_weight_size() const {
    return _wgt_size;
}
