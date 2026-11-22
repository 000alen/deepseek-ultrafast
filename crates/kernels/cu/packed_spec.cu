#include "common.cuh"

struct Meta {
    int pos;
    int m;
    int tokens[16];
};

struct DraftLane {
    Meta* meta;
    const int* proposal;
    const int* verdict;
    Meta input;
};
static_assert(sizeof(DraftLane) == 96);

extern "C" __global__ void packed_spec_draft_meta(const DraftLane* lanes) {
    pdl_wait();
    if (threadIdx.x == 0) *lanes[blockIdx.x].meta = lanes[blockIdx.x].input;
}

extern "C" __global__ void packed_spec_verify_meta(const DraftLane* lanes, int width) {
    pdl_wait();
    const auto& lane = lanes[blockIdx.x];
    if (threadIdx.x == 0) {
        lane.meta->m = width;
        for (int row = 0; row < width; ++row) lane.meta->tokens[row] = lane.proposal[row];
    }
}

extern "C" __global__ void packed_spec_result(const DraftLane* lanes, int width, int vocab, int* results) {
    pdl_wait();
    if (threadIdx.x != 0) return;
    const auto& lane = lanes[blockIdx.x];
    const int accepted = lane.verdict[0], sample = lane.verdict[1];
    auto* result = results + (size_t)blockIdx.x * (width + 2);
    result[0] = accepted;
    result[1] = sample;
    bool valid = accepted >= 0 && accepted < width && sample >= 0 && sample < vocab
        && lane.proposal[0] == lane.input.tokens[0];
    for (int row = 0; row < width; ++row) {
        const int token = lane.proposal[row];
        result[2 + row] = token;
        if (row <= accepted) valid &= token >= 0 && token < vocab;
    }
    // The KV append kernels may launch over the full verifier width, but read
    // the committed prefix length from this per-lane Meta before writing.
    lane.meta->m = valid ? accepted + 1 : 0;
}
