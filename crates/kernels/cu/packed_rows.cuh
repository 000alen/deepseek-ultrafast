#pragma once
#include "common.cuh"

// All addresses are rank-local. KV addresses stay stable as a slot grows and shrinks.
struct PackedLayer {
    bf16* window;
    bf16* compressed;
    bf16* keys;
    uint8_t* keys_fp4;
    uint8_t* key_scales;
    float* kv_history;
    float* score_history;
    int ring_rows;
    int history_rows;
};
struct PackedRow {
    Meta* meta;
    int* history;
    void* sampling;
    const PackedLayer* layers;
    uint8_t* candidate_mask;
    int* candidate_ids;
    int local;
    int position;
    int anchor;
    int pad;
};
static_assert(sizeof(PackedLayer) == 64 && sizeof(PackedRow) == 64);
