#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "../common_function.hpp"
#include "../operations/Operation.h"


#define LAYER(i) ("layer" + std::to_string(i))

namespace BlockType {
extern std::string Attention;
extern std::string FeedForward;
}  // namespace BlockType

namespace OperationType {
    extern std::string LayerNorm;
    extern std::string RMSNorm;
    extern std::string QKVGen;
    extern std::string QGen;
    extern std::string KGen;
    extern std::string VGen;
    extern std::string Projection;
    extern std::string FullyConnected1;
    extern std::string FullyConnected2;
    extern std::string GateProj;
    extern std::string UpProj;
    extern std::string DownProj;
    extern std::string QKVSplit;
    extern std::string QKGEMM;
    extern std::string SoftMax;
    extern std::string SVGEMM;

    extern std::string QKGEMV;
    extern std::string SVGEMV;
    extern std::string QKPIMGEMV;
    extern std::string SVPIMGEMV;

    extern std::string LsVGEMM;
    extern std::string AReshape;
    extern std::string Residual;
    extern std::string Gelu;
    extern std::string SiLU;
    extern std::string Mul;
    extern std::string BatchSplit;
    extern std::string BatchConcat;

    extern std::string KCacheConcat;
    extern std::string VCacheConcat;
    extern std::string VConcat;

    extern std::string PIMGEMVSoftmax;
    extern std::string PIMGEMVAdd;
    extern std::string Microbench;
    extern std::string NeuPIMSLogitSoftmax;
    extern std::string Attention;
    extern std::string NeuPIMSAttend;
    extern std::string FusedMHA;
    extern std::string PIMGEMV;
}  // namespace OperationType

namespace ParameterType {
extern std::string Weight;
extern std::string Bias;
}  // namespace ParameterType

class Model {
   public:
    Model(const SysConfig& config, std::string name);
    Model(const Model &model);
    ~Model();

    void init_my_params();
    Ptr<MyTensor> find_my_tensor(std::string name);
    std::vector<Ptr<MyTensor>> get_my_params(int layer_idx, std::string block_type, std::string operation_type);
    std::shared_ptr<MyTensor> create_my_weight(std::string name, std::vector<uint32_t> dims);

    void finish_operation(uint32_t id);
    void finish_operation_tile(uint32_t id, Tile &tile);

    void find_executable_node(std::shared_ptr<MyTensor> tensor);

    std::string get_name() { return _name; }
    uint32_t get_id() { return _root_node_id; }
    std::shared_ptr<MyTensor> get_input_tensor() { return _input_tensor; }
    std::vector<std::shared_ptr<Operation>> get_executable_operations();
    uint64_t get_weight_size() const;

    std::unordered_map<InferRequest*, std::unordered_map<std::string, Ptr<MyTensor>>> _stored_KVCache;

   private:
    std::string _name;
    std::string _input_name;
    std::vector<uint32_t> _input_dim;
    uint32_t _root_node_id;
    std::shared_ptr<MyTensor> _input_tensor;
    std::map<uint32_t, std::shared_ptr<Operation>> _operation_map;
    std::map<uint32_t, std::shared_ptr<MyTensor>> _tensor_map;
    std::unordered_map<std::string, Ptr<MyTensor>> _my_wgt_map;
    std::vector<std::shared_ptr<Operation>> _executable_operations;
    const SysConfig& _config;
    bool _is_decode{};
    uint64_t _wgt_size;  // bytes


    bool check_exist_in_executable(uint32_t id);
};

#endif
