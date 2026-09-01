#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../common_function.hpp"
#include "RequestGenerator.h"

uint32_t generate_rid();
class Client {
   public:
    Client(const SysConfig&);
    void cycle();

    bool running();
    bool has_request();
    bool request_can_issue();
    std::shared_ptr<InferRequest> pop_request();
    void receive_response(std::shared_ptr<InferRequest> response);
    void write_request_stats() const;
    virtual void set_client_cycle(cycle_type cycle);
    cycle_type current_cycle() const { return _cycles; }

   private:
    const SysConfig& _config;
    cycle_type _cycles;
    cycle_type _valid_request_cycle;
    cycle_type _need_wait_cycles;

    bool gen_request;
    bool gen_random_request;
    u_int32_t gen_request_input_size;
    u_int32_t gen_request_output_size;

    uint32_t _total_cnt;
    uint32_t _issued_cnt;
    uint32_t _completed_cnt;
    bool _completion_logged;

    cycle_type _request_interval;  // send a request per (core_freq/qps) cycles
    std::queue<std::shared_ptr<InferRequest>> _waiting_queue;
    std::vector<std::shared_ptr<InferRequest>> _completed_requests;

    std::poisson_distribution<> request_interval_random;

    uint32_t request_id;

    /* Random generate from uniform d (input, output size) [min, max)*/
    int _imin;
    int _imax;
    int _omin;
    int _omax;
    int rand_input_size();
    int rand_output_size();

};
