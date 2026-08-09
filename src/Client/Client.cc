#include "Client.h"

Client::Client(const SysConfig& config)
    : _config(config),
      _cycles(0),
      _valid_request_cycle(0),
      _issued_cnt(0),
      _completed_cnt(0),
      _completion_logged(false),
      _need_wait_cycles(0) {  // 30
    // arguments:
    // - request_interval (mean),
    // - total number of requests
    // - request size (input,output)

    gen_request = _config.gen_request;
    gen_random_request = _config.gen_random_request;
    gen_request_input_size = _config.gen_request_input_size;
    gen_request_output_size = _config.gen_request_output_size;

    request_id = 0;

    if (gen_request) {
        spdlog::info("Generation request size: {}", gen_request_input_size);
        _total_cnt = _config.gen_request_count;  // config by json file
    }
    else {
        uint32_t answer_index = 1;
        RequestGenerator::init(config.request_dataset_path, answer_index);
        _total_cnt = RequestGenerator::get_total_req_cnt();
        spdlog::info("Client total request cnt: {}", _total_cnt);
    }

    //
    _request_interval = _config.request_interval;
    std::poisson_distribution<> request_d(_request_interval);
    request_interval_random = request_d;

    // todo: get from config
    _imin = 128;
    _imax = 256;
    _omin = 256;
    _omax = 512;
}

int Client::rand_input_size() { return rand() % (_imax - _imin) + _imin; }
int Client::rand_output_size() { return rand() % (_omax - _omin) + _omin; }

void Client::cycle() {
    uint32_t idle_cycles = _cycles - _valid_request_cycle;

    uint32_t input_size;
    uint32_t output_size;

    // Ad Request
    if (request_id < _total_cnt and _cycles >= _valid_request_cycle) {
        if (!gen_request) {
            // from RequestGenerator
            std::pair<uint32_t, uint32_t> input_output_size;
            if (RequestGenerator::has_data()) {
                input_output_size = RequestGenerator::get_qa_length();  // 当前RequestGenerator
            }
            else {
                spdlog::info("RequestGenerator has no data!");
                // exit(-1);
            }
            input_size = input_output_size.first;
            output_size = gen_request_output_size;  // generate a random output size ;
        }
        else {
            if (gen_random_request) {
                input_size = rand_input_size();
                output_size = rand_output_size();
            }
            else {
                input_size = gen_request_input_size;
                output_size = gen_request_output_size;
            }
        }

        std::shared_ptr<InferRequest> request =
            std::make_shared<InferRequest>(InferRequest{.id = request_id, // request_index
                                                        .arrival_cycle = _cycles,  // the Operation cycle
                                                        .completed_cycle = 0, // request finished cycle
                                                        .input_size = input_size, // input token length size
                                                        .output_size = output_size, // output token length
                                                        .is_initiated = false,
                                                        .generated = 0,  // tokens generated
                                                        .channel = 0});
        _waiting_queue.push(request);
        spdlog::info("Request #{} is generated at client time {}, input size:{}, output size:{}", request_id, _cycles,  input_size, output_size);

        _need_wait_cycles = _request_interval;
        _valid_request_cycle = _cycles + _need_wait_cycles;
        request_id ++;
    }

    if (!_completion_logged && _completed_cnt == _total_cnt) {
        spdlog::info("All Request of Client has completed!");
        _completion_logged = true;
    }
    _cycles++;
}

bool Client::running() {
    return _completed_cnt < _total_cnt;  // FIXME: comment
    return false;
}

bool Client::has_request() { return !_waiting_queue.empty();}  // has request, has no new trace input

std::shared_ptr<InferRequest> Client::pop_request() {
    std::shared_ptr<InferRequest> top = _waiting_queue.front();
    _waiting_queue.pop();
    _issued_cnt++;
    spdlog::info("Client issued Request cnt: {} total: {}, at time: {}", _issued_cnt, _total_cnt, _cycles);
    // spdlog::info("Need wait: {}, next valid request cycle: {}", _need_wait_cycles, _valid_request_cycle);
    return top;
}

void Client::receive_response(std::shared_ptr<InferRequest> response) {
    // ast(response->generated == response->output_size);
    response->completed_cycle = _cycles;
    _completed_cnt++;

    spdlog::info("Client Receive response From Scheduler! spend_cycles: {}", response->completed_cycle - response->arrival_cycle);
    if (!_completion_logged && _completed_cnt == _total_cnt) {
        spdlog::info("All Request of Client has completed!");
        _completion_logged = true;
    }

    // todo stat.
    // delete response;
}

void Client::set_client_cycle(cycle_type cycle) {
    _cycles = cycle;
}

uint32_t generate_rid() {
    static uint32_t rid{0};
    return rid++;
}
