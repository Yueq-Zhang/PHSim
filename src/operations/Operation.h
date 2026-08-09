#pragma once

#include "../common_function.hpp"
// #include "../Core/Mapping.h"

#include "../tensor/MyTensor.hpp"

class Model;
class OpParser;

// Graph Node
//继承 std::enable_shared_from_this<Operation> 后，Operation 类内部可通过 shared_from_this() 方法获取指向当前对象的 std::shared_ptr
class Operation : public std::enable_shared_from_this<Operation> {
   public:
    virtual ~Operation() = default;

    //Operation(MappingTable mapping_table);
    Operation(std::string name);
    Operation(const Operation &operation);

    void initialize(const SysConfig& config) {}

    virtual void set_finish();

    virtual std::string get_name() { return _name; }
    virtual std::string get_optype() { return _optype; }
    virtual uint32_t get_id() { return _id; }
    virtual uint32_t num_inputs() { return _inputs.size(); }
    virtual std::vector<std::shared_ptr<BTensor>> get_inputs() { return _inputs; }
    virtual uint32_t num_outputs() { return _outputs.size(); }
    virtual std::vector<std::shared_ptr<Operation>> get_child_nodes();
    virtual std::deque<Tile> get_tiles();
    virtual bool check_executable();
    virtual std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs);
    void set_as_parent_tensor(std::vector<Ptr<BTensor>> inputs);
    // 函数重载
    void set_as_parent_tensor(std::vector<Ptr<MyTensor>> inputs);  // New added
    virtual std::vector<Ptr<MyTensor>> get_my_outputs(std::vector<Ptr<MyTensor>> inputs, TensorType output_tensor_type=TensorType::ACT);
    virtual std::vector<Ptr<MyTensor>> kvcache_append(std::vector<Ptr<MyTensor>> inputs, std::vector<Ptr<MyTensor>> kvcaches, TensorType output_tensor_type=TensorType::ACT);

    virtual std::vector<uint32_t> get_inner_loop() {return _inner_loop; }
    virtual std::vector<uint32_t> get_outer_loop() {return _outer_loop; }

    void set_apply_rope(bool apply_rope){_apply_rope = apply_rope;}

    void set_apply_rope() {
        if (is_llama_model_name(Config::system_config.model_name)) {
            _apply_rope = true;
        }
        else {
            _apply_rope = false;
        }
    }

    bool check_finish() { return _finish; };

    void reduce_tile(Tile &tile);
    OperationStat get_stat() { return _stat; };
    std::string repr();

   protected:
    // virtual void initialize_instructions(Tile &tile, Mapping mapping) {}

    static const uint32_t _NO_OPERAND = 0;
    static const uint32_t _INPUT_OPERAND = 100;
    static const uint32_t _OUTPUT_OPERAND = 200;
    uint32_t _id;
    std::string _name;
    std::string _optype;
    const SysConfig& _config = Config::system_config;
    std::vector<Ptr<BTensor>> _inputs;
    std::vector<Ptr<BTensor>> _outputs;

    std::vector<uint32_t> _inner_loop;
    std::vector<uint32_t> _outer_loop;

    std::vector<Ptr<MyTensor>> _my_inputs;
    std::vector<Ptr<MyTensor>> _my_weights;
    std::vector<Ptr<MyTensor>> _my_outputs;

    uint32_t _num_inputs = 1;
    uint32_t _num_outputs = 1;
    bool _apply_rope = false;

    uint32_t _batch_size;
    std::vector<uint32_t> _m_batch_dim;

    std::map<std::string, std::string> _attributes;
    std::deque<Tile> _tiles;
    std::vector<std::vector<std::vector<addr_type>>> _weight_addrs;
    std::vector<std::vector<std::vector<std::vector<addr_type>>>> _input_addrs;
    std::vector<std::vector<std::vector<std::vector<addr_type>>>> _output_addrs;

    int Ndim;    // Batch dimension of activation tensor (commoly 0)
    int Hdim;    // Height dimension of activation tensor
    int Wdim;    // Width dimension of activation tensor
    int Cdim;    // Channel dimension of activation tensor
    int Cdim_w;  // Channel dimension of weight tensor
    int Mdim;    // Output channel dimension of weight tensor
    int Sdim;    // Height dimension of weight tensor
    int Rdim;    // Width dimension of weight tensor

    OpStat _op_stat;
    OperationStat _stat;

    bool _finish;
    friend Model;
    addr_type _spad_addr;
    addr_type _acc_spad_addr;

    std::pair<addr_type, uint32_t> allocate_sram_addr(uint32_t size, bool accum);

    bool defer_decode_pruning_compile() const {
        return _config.decode_pruning_enabled &&
               _config.decode_pruning_compile_context &&
               _name.find(".ffn.") != std::string::npos;
    }
};