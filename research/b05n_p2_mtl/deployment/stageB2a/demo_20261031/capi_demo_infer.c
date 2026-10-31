#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rknn_api.h"

#define MAX_OUT 15
#define MAX_DET 4000
#define CONF 0.25f
#define NMS_IOU 0.45f

typedef struct { float x1, y1, x2, y2, score; int cls; } Det;
typedef struct { uint32_t n_elems, dims[4]; float scale; int32_t zp; int type; int fmt; char name[64]; } Attr;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static float qval(const void *buf, Attr *a, uint32_t idx) {
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
static int decode_det(void **bufs, Attr *attrs, Det *out) {
    Det cand[MAX_DET]; int nc = 0;
    for (int s = 0; s < 4; s++) {
        Attr *ra = &attrs[s * 3], *ca = &attrs[s * 3 + 1];
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
            if (rx <= cx || by <= cy || nc >= MAX_DET) continue;
            cand[nc++] = (Det){cx, cy, rx, by, bs, best};
        }
    }
    int n = 0;
    for (int c = 0; c < 10; c++) {
        Det tmp[MAX_DET]; int m = 0;
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

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s model.rknn inputs.bin frames det.jsonl masks.bin\n", argv[0]); return 1; }
    int frames = atoi(argv[3]);
    FILE *mf = fopen(argv[1], "rb"); if (!mf) { perror("model"); return 1; }
    fseek(mf, 0, SEEK_END); long ms = ftell(mf); fseek(mf, 0, SEEK_SET);
    unsigned char *model = malloc(ms); if (fread(model, 1, ms, mf) != (size_t)ms) return 1; fclose(mf);
    FILE *inf = fopen(argv[2], "rb"); if (!inf) { perror("inputs"); return 1; }
    long is = 512L * 288L * 3L * frames;
    unsigned char *inputs = malloc(is); if (fread(inputs, 1, is, inf) != (size_t)is) return 1; fclose(inf);
    FILE *df = fopen(argv[4], "w"); if (!df) { perror("det"); return 1; }
    FILE *mfout = fopen(argv[5], "wb"); if (!mfout) { perror("masks"); return 1; }

    rknn_context ctx;
    if (rknn_init(&ctx, model, ms, 0, NULL)) { fprintf(stderr, "init failed\n"); return 1; }
    rknn_input_output_num io; rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (io.n_output < 15) { fprintf(stderr, "expected 15 outputs\n"); return 1; }
    Attr attrs[MAX_OUT];
    for (uint32_t i = 0; i < io.n_output; i++) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        attrs[i] = (Attr){a.n_elems, {a.dims[0], a.dims[1], a.dims[2], a.dims[3]}, a.scale, a.zp, a.type, a.fmt, ""};
    }
    int H = attrs[12].dims[2], W = attrs[12].dims[3], HW = H * W;
    fprintf(stderr, "seg=%dx%d frames=%d\n", W, H, frames);
    unsigned char *masks = malloc((size_t)HW * 3);
    Det dets[MAX_DET];
    double t_run = 0, t_get = 0, t_det = 0, t_seg = 0, t_all = 0;
    int total_det = 0;
    double wall0 = now_ms();
    for (int f = 0; f < frames; f++) {
        double t0 = now_ms();
        rknn_input in; memset(&in, 0, sizeof(in));
        in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
        in.size = 512 * 288 * 3; in.buf = inputs + (size_t)f * 512 * 288 * 3;
        rknn_inputs_set(ctx, 1, &in);
        rknn_output outs[15]; memset(outs, 0, sizeof(outs));
        for (int i = 0; i < 15; i++) { outs[i].index = i; outs[i].want_float = 0; }
        double a = now_ms(); rknn_run(ctx, NULL); double b = now_ms();
        rknn_outputs_get(ctx, 15, outs, NULL); double c = now_ms();
        void *bufs[15]; for (int i = 0; i < 15; i++) bufs[i] = outs[i].buf;
        int nd = decode_det(bufs, attrs, dets); double d = now_ms();
        for (int h = 0; h < 3; h++) {
            const int8_t *p = (const int8_t *)outs[12 + h].buf;
            int C = h == 0 ? 9 : 3;
            unsigned char *dst = masks + (size_t)h * HW;
            for (int i = 0; i < HW; i++) {
                int best = 0; int8_t bv = p[i];
                for (int cc = 1; cc < C; cc++) {
                    int8_t v = p[(size_t)cc * HW + i];
                    if (v > bv) { bv = v; best = cc; }
                }
                dst[i] = (unsigned char)best;
            }
        }
        double e = now_ms();
        fwrite(masks, 1, (size_t)HW * 3, mfout);
        fprintf(df, "{\"frame\":%d,\"detections\":[", f);
        for (int i = 0; i < nd; i++) {
            fprintf(df, "%s[%d,%.4f,%.2f,%.2f,%.2f,%.2f]", i ? "," : "",
                    dets[i].cls, dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
        }
        fprintf(df, "]}\n");
        rknn_outputs_release(ctx, 15, outs);
        double z = now_ms();
        t_run += b - a; t_get += c - b; t_det += d - c; t_seg += e - d; t_all += z - t0;
        total_det += nd;
    }
    double wall = (now_ms() - wall0) / 1000.0;
    fclose(df); fclose(mfout);
    printf("{\"frames\":%d,\"detections\":%d,\"seg_hw\":%d,\"mean_run_ms\":%.3f,\"mean_get_ms\":%.3f,\"mean_det_ms\":%.3f,\"mean_seg_ms\":%.3f,\"mean_total_ms\":%.3f,\"wall_s\":%.3f,\"fps\":%.3f}\n",
           frames, total_det, HW, t_run / frames, t_get / frames, t_det / frames, t_seg / frames, t_all / frames, wall, frames / wall);
    rknn_destroy(ctx);
    free(model); free(inputs); free(masks);
    return 0;
}
