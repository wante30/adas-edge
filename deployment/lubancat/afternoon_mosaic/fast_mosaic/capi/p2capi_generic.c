#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rknn_api.h"

#define MAX_OUT 15
#define MAX_DET 1000
#define MAX_CAND 4000
#define CONF 0.25f
#define NMS_IOU 0.45f

typedef struct { float x1, y1, x2, y2, score; int cls; } Det;
typedef struct {
    uint32_t n_elems, dims[4];
    float scale;
    int32_t zp;
    int type;
    int fmt;
    char name[64];
} Attr;

typedef struct {
    rknn_context ctx;
    int n_outputs;
    int n_det;          /* 9 (BDD 3-scale) or 12 (P2 4-scale) */
    int n_scales;       /* 3 or 4 */
    int n_classes;      /* 10 */
    Attr attrs[MAX_OUT];
    int reg_idx[4];
    int cls_idx[4];
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

/* DFL decode for one scale; attrs are already bound to this scale. */
static int decode_scale(void **bufs, const P2Context *c, int s, Det *cand, int nc) {
    const Attr *ra = &c->attrs[c->reg_idx[s]];
    const Attr *ca = &c->attrs[c->cls_idx[s]];
    const void *reg_buf = bufs[c->reg_idx[s]];
    const void *cls_buf = bufs[c->cls_idx[s]];
    int H = (int)ra->dims[2], W = (int)ra->dims[3];
    int HW = H * W;
    int ncls = c->n_classes;
    if (ncls > 10) ncls = 10;
    float sx = 512.0f / (float)W;
    float sy = 288.0f / (float)H;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            int best = 0;
            float bs = -1.0f;
            for (int k = 0; k < ncls; k++) {
                float v = qval(cls_buf, ca, (uint32_t)k * HW + idx);
                if (v > bs) { bs = v; best = k; }
            }
            if (bs < CONF) continue;
            float d[4];
            for (int side = 0; side < 4; side++) {
                float m = -1e30f;
                for (int b = 0; b < 16; b++) {
                    float v = qval(reg_buf, ra, (uint32_t)(side * 16 + b) * HW + idx);
                    if (v > m) m = v;
                }
                float sum = 0.0f, es = 0.0f;
                for (int b = 0; b < 16; b++) {
                    float v = qval(reg_buf, ra, (uint32_t)(side * 16 + b) * HW + idx);
                    float e = expf(v - m);
                    es += e;
                    sum += (float)b * e;
                }
                d[side] = sum / es;
            }
            float cx = ((float)x + 0.5f - d[0]) * sx;
            float cy = ((float)y + 0.5f - d[1]) * sy;
            float rx = ((float)x + 0.5f + d[2]) * sx;
            float by = ((float)y + 0.5f + d[3]) * sy;
            if (rx <= cx || by <= cy || nc >= MAX_CAND) continue;
            cand[nc++] = (Det){cx, cy, rx, by, bs, best};
        }
    }
    return nc;
}

static int decode_det(void **bufs, const P2Context *c, Det *out) {
    Det cand[MAX_CAND];
    int nc = 0;
    for (int s = 0; s < c->n_scales; s++) {
        nc = decode_scale(bufs, c, s, cand, nc);
    }
    int n = 0;
    for (int cls = 0; cls < c->n_classes && cls < 10; cls++) {
        Det tmp[MAX_CAND];
        int m = 0;
        for (int i = 0; i < nc; i++) {
            if (cand[i].cls == cls) tmp[m++] = cand[i];
        }
        qsort(tmp, (size_t)m, sizeof(Det), cmp_det);
        int kept = 0;
        for (int i = 0; i < m; i++) {
            int ok = 1;
            for (int j = 0; j < kept; j++) {
                if (iou(tmp[i], out[n - kept + j]) > NMS_IOU) { ok = 0; break; }
            }
            if (ok && n < MAX_DET) {
                out[n++] = tmp[i];
                kept++;
            }
        }
    }
    return n;
}

static int query_attrs(P2Context *c) {
    rknn_input_output_num io;
    if (rknn_query(c->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io))) return -1;
    if (io.n_output != 9 && io.n_output != 12 && io.n_output != 15) return -1;
    c->n_outputs = (int)io.n_output;
    c->n_det = c->n_outputs >= 15 ? 12 : c->n_outputs;
    if (c->n_det % 3) return -1;
    c->n_scales = c->n_det / 3;

    for (int i = 0; i < c->n_outputs; i++) {
        rknn_tensor_attr a;
        memset(&a, 0, sizeof(a));
        a.index = (uint32_t)i;
        if (rknn_query(c->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a))) return -1;
        c->attrs[i].n_elems = a.n_elems;
        c->attrs[i].dims[0] = a.dims[0];
        c->attrs[i].dims[1] = a.dims[1];
        c->attrs[i].dims[2] = a.dims[2];
        c->attrs[i].dims[3] = a.dims[3];
        c->attrs[i].scale = a.scale;
        c->attrs[i].zp = a.zp;
        c->attrs[i].type = a.type;
        c->attrs[i].fmt = a.fmt;
        snprintf(c->attrs[i].name, sizeof(c->attrs[i].name), "%s", a.name);
    }

    /* Bind reg/cls by tensor name when available; fall back to the
       documented per-scale order (reg, cls, cls_sum). */
    for (int s = 0; s < c->n_scales; s++) {
        int base = s * 3;
        c->reg_idx[s] = base;
        c->cls_idx[s] = base + 1;
        int reg_found = -1, cls_found = -1;
        for (int j = 0; j < 3; j++) {
            const char *name = c->attrs[base + j].name;
            if (strstr(name, "reg")) reg_found = base + j;
            else if (strstr(name, "cls_sum")) { /* score sum, not class */ }
            else if (strstr(name, "cls")) cls_found = base + j;
        }
        if (reg_found >= 0 && cls_found >= 0) {
            c->reg_idx[s] = reg_found;
            c->cls_idx[s] = cls_found;
        }
    }

    const Attr *cls0 = &c->attrs[c->cls_idx[0]];
    c->n_classes = (int)cls0->dims[1];
    if (c->n_classes <= 0 || c->n_classes > 10) return -1;

    if (c->n_outputs >= 15) {
        for (int i = 0; i < 3; i++) {
            c->seg_attrs[i] = c->attrs[12 + i];
        }
        c->seg_h = (int)c->seg_attrs[0].dims[2];
        c->seg_w = (int)c->seg_attrs[0].dims[3];
        c->seg_hw = c->seg_h * c->seg_w;
        if (c->seg_hw <= 0) return -1;
    }
    return 0;
}

void *p2_create(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long size = ftell(f);
    if (size <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    unsigned char *model = malloc((size_t)size);
    if (!model || fread(model, 1, (size_t)size, f) != (size_t)size) {
        fclose(f); free(model); return NULL;
    }
    fclose(f);
    P2Context *c = calloc(1, sizeof(P2Context));
    if (!c) { free(model); return NULL; }
    if (rknn_init(&c->ctx, model, (uint32_t)size, 0, NULL)) {
        free(model); free(c); return NULL;
    }
    free(model);
    if (query_attrs(c)) {
        rknn_destroy(c->ctx); free(c); return NULL;
    }
    return c;
}

int p2_output_count(void *handle) {
    P2Context *c = (P2Context *)handle;
    return c ? c->n_outputs : 0;
}

int p2_scale_count(void *handle) {
    P2Context *c = (P2Context *)handle;
    return c ? c->n_scales : 0;
}

int p2_detect(void *handle, const uint8_t *rgb, P2Result *out) {
    P2Context *c = (P2Context *)handle;
    if (!c || !rgb || !out) return -1;
    out->count = 0;
    double t0 = now_ms();
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = 512 * 288 * 3;
    in.buf = (void *)rgb;
    if (rknn_inputs_set(c->ctx, 1, &in)) return -2;
    rknn_output outs[MAX_OUT];
    memset(outs, 0, sizeof(outs));
    for (int i = 0; i < c->n_det; i++) {
        outs[i].index = (uint32_t)i;
        outs[i].want_float = 0;
    }
    double t1 = now_ms();
    if (rknn_run(c->ctx, NULL)) return -3;
    double t2 = now_ms();
    if (rknn_outputs_get(c->ctx, (uint32_t)c->n_det, outs, NULL)) return -4;
    double t3 = now_ms();
    void *bufs[MAX_OUT];
    for (int i = 0; i < c->n_det; i++) bufs[i] = outs[i].buf;
    Det dets[MAX_DET];
    int n = decode_det(bufs, c, dets);
    double t4 = now_ms();
    rknn_outputs_release(c->ctx, (uint32_t)c->n_det, outs);
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
    if (!c || !c->seg_hw) {
        if (h) *h = 0;
        if (w) *w = 0;
        return;
    }
    if (h) *h = c->seg_h;
    if (w) *w = c->seg_w;
}

int p2_detect_mtl(void *handle, const uint8_t *rgb, P2Result *out, uint8_t *masks) {
    P2Context *c = (P2Context *)handle;
    if (!c || !rgb || !out || !masks || c->n_outputs < 15) return -1;
    out->count = 0;
    double t0 = now_ms();
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = 512 * 288 * 3;
    in.buf = (void *)rgb;
    if (rknn_inputs_set(c->ctx, 1, &in)) return -2;
    rknn_output outs[MAX_OUT];
    memset(outs, 0, sizeof(outs));
    for (int i = 0; i < c->n_outputs; i++) {
        outs[i].index = (uint32_t)i;
        outs[i].want_float = 0;
    }
    double t1 = now_ms();
    if (rknn_run(c->ctx, NULL)) return -3;
    double t2 = now_ms();
    if (rknn_outputs_get(c->ctx, (uint32_t)c->n_outputs, outs, NULL)) return -4;
    double t3 = now_ms();
    void *bufs[MAX_OUT];
    for (int i = 0; i < c->n_outputs; i++) bufs[i] = outs[i].buf;
    Det dets[MAX_DET];
    int n = decode_det(bufs, c, dets);
    int HW = c->seg_hw;
    for (int h = 0; h < 3; h++) {
        const int8_t *p = (const int8_t *)outs[12 + h].buf;
        int C = h == 0 ? 9 : 3;
        uint8_t *dst = masks + (size_t)h * HW;
        for (int i = 0; i < HW; i++) {
            int best = 0;
            int8_t bv = p[i];
            for (int cc = 1; cc < C; cc++) {
                int8_t v = p[(size_t)cc * HW + i];
                if (v > bv) { bv = v; best = cc; }
            }
            dst[i] = (uint8_t)best;
        }
    }
    double t4 = now_ms();
    rknn_outputs_release(c->ctx, (uint32_t)c->n_outputs, outs);
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
