#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rknn_api.h"

#define MAX_DET 1000
#define CONF 0.25f
#define NMS_IOU 0.45f

typedef struct { float x1, y1, x2, y2, score; int cls; } Det;
typedef struct { uint32_t n_elems, dims[4]; float scale; int32_t zp; int type; } Attr;

typedef struct {
    rknn_context ctx;
    int n_outputs;
    Attr attrs[12];
    int seg_h;
    int seg_w;
    int seg_hw;
    Attr seg_attrs[3];
} P2Context;

typedef struct {
    int count;
    float rknn_ms;
    float decode_ms;
    float total_ms;
    float data[MAX_DET][6];
} P2Result;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static float qval(const void *buf, const Attr *a, uint32_t idx) {
    return (((const int8_t *)buf)[idx] - a->zp) * a->scale;
}
static int cmp_det(const void *a, const void *b) {
    float x = ((Det *)a)->score, y = ((Det *)b)->score;
    return x < y ? 1 : (x > y ? -1 : 0);
}
static float iou(Det a, Det b) {
    float x1 = fmaxf(a.x1, b.x1), y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2), y2 = fminf(a.y2, b.y2);
    float inter = fmaxf(0, x2 - x1) * fmaxf(0, y2 - y1);
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1), bb = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / fmaxf(aa + bb - inter, 1e-9f);
}

static int decode_det(void **bufs, const Attr *attrs, Det *out) {
    Det cand[4000]; int nc = 0;
    for (int s = 0; s < 4; s++) {
        const Attr *ra = &attrs[s * 3], *ca = &attrs[s * 3 + 1];
        int H = ra->dims[2], W = ra->dims[3];
        float sx = 512.0f / W, sy = 288.0f / H;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            int idx = y * W + x; int best = 0; float bs = -1;
            for (int c = 0; c < 10; c++) {
                float v = qval(bufs[s * 3 + 1], ca, (uint32_t)c * H * W + idx);
                if (v > bs) { bs = v; best = c; }
            }
            if (bs < CONF) continue;
            float d[4];
            for (int side = 0; side < 4; side++) {
                float m = -1e30f;
                for (int b = 0; b < 16; b++) {
                    float v = qval(bufs[s * 3], ra, (uint32_t)(side * 16 + b) * H * W + idx);
                    if (v > m) m = v;
                }
                float sum = 0, es = 0;
                for (int b = 0; b < 16; b++) {
                    float v = qval(bufs[s * 3], ra, (uint32_t)(side * 16 + b) * H * W + idx);
                    float e = expf(v - m); es += e; sum += b * e;
                }
                d[side] = sum / es;
            }
            float cx = (x + 0.5f - d[0]) * sx, cy = (y + 0.5f - d[1]) * sy;
            float rx = (x + 0.5f + d[2]) * sx, by = (y + 0.5f + d[3]) * sy;
            if (rx <= cx || by <= cy || nc >= 4000) continue;
            cand[nc++] = (Det){cx, cy, rx, by, bs, best};
        }
    }
    int n = 0;
    for (int c = 0; c < 10; c++) {
        Det tmp[4000]; int m = 0;
        for (int i = 0; i < nc; i++) if (cand[i].cls == c) tmp[m++] = cand[i];
        qsort(tmp, m, sizeof(Det), cmp_det);
        int kept = 0;
        for (int i = 0; i < m; i++) {
            int ok = 1;
            for (int j = 0; j < kept; j++) {
                if (iou(tmp[i], out[n - kept + j]) > NMS_IOU) { ok = 0; break; }
            }
            if (ok && n < MAX_DET) { out[n++] = tmp[i]; kept++; }
        }
    }
    return n;
}

void *p2_create(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *model = malloc(size);
    if (!model || fread(model, 1, size, f) != (size_t)size) { fclose(f); free(model); return NULL; }
    fclose(f);
    P2Context *c = calloc(1, sizeof(P2Context));
    if (!c) { free(model); return NULL; }
    if (rknn_init(&c->ctx, model, size, 0, NULL)) { free(model); free(c); return NULL; }
    free(model);
    rknn_input_output_num io;
    if (rknn_query(c->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) || io.n_output < 12) {
        rknn_destroy(c->ctx); free(c); return NULL;
    }
    c->n_outputs = (int)io.n_output;
    for (int i = 0; i < 12; i++) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = i;
        if (rknn_query(c->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a))) {
            rknn_destroy(c->ctx); free(c); return NULL;
        }
        c->attrs[i] = (Attr){a.n_elems, {a.dims[0], a.dims[1], a.dims[2], a.dims[3]},
                             a.scale, a.zp, a.type};
    }
    if (io.n_output >= 15) {
        for (int i = 0; i < 3; i++) {
            rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = 12 + i;
            if (rknn_query(c->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a))) {
                rknn_destroy(c->ctx); free(c); return NULL;
            }
            c->seg_attrs[i] = (Attr){a.n_elems, {a.dims[0], a.dims[1], a.dims[2], a.dims[3]},
                                     a.scale, a.zp, a.type};
        }
        c->seg_h = (int)c->seg_attrs[0].dims[2];
        c->seg_w = (int)c->seg_attrs[0].dims[3];
        c->seg_hw = c->seg_h * c->seg_w;
    }
    return c;
}

int p2_detect(void *handle, const uint8_t *rgb, P2Result *out) {
    P2Context *c = (P2Context *)handle;
    if (!c || !rgb || !out) return -1;
    out->count = 0;
    double t0 = now_ms();
    rknn_input in; memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    in.size = 512 * 288 * 3; in.buf = (void *)rgb;
    if (rknn_inputs_set(c->ctx, 1, &in)) return -2;
    rknn_output outs[12]; memset(outs, 0, sizeof(outs));
    for (int i = 0; i < 12; i++) { outs[i].index = i; outs[i].want_float = 0; }
    double t1 = now_ms();
    if (rknn_run(c->ctx, NULL)) return -3;
    double t2 = now_ms();
    if (rknn_outputs_get(c->ctx, 12, outs, NULL)) return -4;
    double t3 = now_ms();
    void *bufs[12];
    for (int i = 0; i < 12; i++) bufs[i] = outs[i].buf;
    Det dets[MAX_DET];
    int n = decode_det(bufs, c->attrs, dets);
    double t4 = now_ms();
    rknn_outputs_release(c->ctx, 12, outs);
    if (n > MAX_DET) n = MAX_DET;
    out->count = n;
    out->rknn_ms = (float)(t2 - t1);
    out->decode_ms = (float)(t4 - t3);
    out->total_ms = (float)(t4 - t0);
    for (int i = 0; i < n; i++) {
        out->data[i][0] = (float)dets[i].cls;
        out->data[i][1] = dets[i].score;
        out->data[i][2] = dets[i].x1;
        out->data[i][3] = dets[i].y1;
        out->data[i][4] = dets[i].x2;
        out->data[i][5] = dets[i].y2;
    }
    return 0;
}

void p2_seg_size(void *handle, int *h, int *w) {
    P2Context *c = (P2Context *)handle;
    if (!c || !c->seg_hw) { if (h) *h = 0; if (w) *w = 0; return; }
    if (h) *h = c->seg_h;
    if (w) *w = c->seg_w;
}

int p2_detect_mtl(void *handle, const uint8_t *rgb, P2Result *out, uint8_t *masks) {
    P2Context *c = (P2Context *)handle;
    if (!c || !rgb || !out || !masks || c->n_outputs < 15) return -1;
    out->count = 0;
    double t0 = now_ms();
    rknn_input in; memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    in.size = 512 * 288 * 3; in.buf = (void *)rgb;
    if (rknn_inputs_set(c->ctx, 1, &in)) return -2;
    rknn_output outs[15]; memset(outs, 0, sizeof(outs));
    for (int i = 0; i < 15; i++) { outs[i].index = i; outs[i].want_float = 0; }
    double t1 = now_ms();
    if (rknn_run(c->ctx, NULL)) return -3;
    double t2 = now_ms();
    if (rknn_outputs_get(c->ctx, 15, outs, NULL)) return -4;
    double t3 = now_ms();
    void *bufs[15];
    for (int i = 0; i < 15; i++) bufs[i] = outs[i].buf;
    Det dets[MAX_DET];
    int n = decode_det(bufs, c->attrs, dets);
    int HW = c->seg_hw;
    for (int h = 0; h < 3; h++) {
        const int8_t *p = (const int8_t *)outs[12 + h].buf;
        int C = h == 0 ? 9 : 3;
        uint8_t *dst = masks + (size_t)h * HW;
        for (int i = 0; i < HW; i++) {
            int best = 0; int8_t bv = p[i];
            for (int cc = 1; cc < C; cc++) {
                int8_t v = p[(size_t)cc * HW + i];
                if (v > bv) { bv = v; best = cc; }
            }
            dst[i] = (uint8_t)best;
        }
    }
    double t4 = now_ms();
    rknn_outputs_release(c->ctx, 15, outs);
    if (n > MAX_DET) n = MAX_DET;
    out->count = n;
    out->rknn_ms = (float)(t2 - t1);
    out->decode_ms = (float)(t4 - t3);
    out->total_ms = (float)(t4 - t0);
    for (int i = 0; i < n; i++) {
        out->data[i][0] = (float)dets[i].cls;
        out->data[i][1] = dets[i].score;
        out->data[i][2] = dets[i].x1;
        out->data[i][3] = dets[i].y1;
        out->data[i][4] = dets[i].x2;
        out->data[i][5] = dets[i].y2;
    }
    return 0;
}

void p2_destroy(void *handle) {
    P2Context *c = (P2Context *)handle;
    if (!c) return;
    rknn_destroy(c->ctx);
    free(c);
}
