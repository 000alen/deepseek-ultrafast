#pragma once
#include "common.cuh"

struct AttnFParams {
    const bf16* q;
    const bf16* win;
    const bf16* comp;
    const int* topk;
    int ktop;
    float scale;
    const float* sink;
    bf16* out;
    const float* cos;
    const float* sin;
    int R;
    int fixed_window;
    int nh;
    int pre;
    int win_floor;
    int ring_rows;
};
static_assert(sizeof(AttnFParams) == 96 && offsetof(AttnFParams, ring_rows) == 92);
