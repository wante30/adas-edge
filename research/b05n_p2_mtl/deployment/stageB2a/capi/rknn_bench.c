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
typedef struct {
    uint32_t n_elems, dims[4];
    float scale;
    int32_t zp;
    int type;
    int fmt;
    char name[64];
} Attr;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { perror("read"); exit(1); }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

static void decode565_rgb(const uint16_t *src, unsigned char *dst, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        uint16_t v = src[i];
        unsigned char r = (unsigned char)(((v >> 11) & 0x1f) * 255 / 31);
        unsigned char g = (unsigned char)(((v >> 5) & 0x3f) * 255 / 63);
        unsigned char b = (unsigned char)((v & 0x1f) * 255 / 31);
        dst[i * 3 + 0] = b; dst[i * 3 + 1] = g; dst[i * 3 + 2] = r;
    }
}

static int x0[512], x1[512];
static float wx[512];
static int sy0[288], sy1[288];
static float wy[288];

static void init_resize(void) {
    for (int x = 0; x < 512; x++) {
        float fx = (x + 0.5f) * 1280 / 512 - 0.5f;
        int a = (int)fx; if (a < 0) a = 0; if (a > 1279) a = 1279;
        int b = a + 1 < 1280 ? a + 1 : a;
        x0[x] = a; x1[x] = b; wx[x] = fx - a; if (wx[x] < 0) wx[x] = 0;
    }
    for (int y = 0; y < 288; y++) {
        float fy = (y + 0.5f) * 720 / 288 - 0.5f;
        int a = (int)fy; if (a < 0) a = 0; if (a > 719) a = 719;
        int b = a + 1 < 720 ? a + 1 : a;
        sy0[y] = a; sy1[y] = b; wy[y] = fy - a; if (wy[y] < 0) wy[y] = 0;
    }
}

static void resize_bgr_to_rgb(const unsigned char *src, unsigned char *dst) {
    for (int y = 0; y < 288; y++) {
        int ya = sy0[y], yb = sy1[y]; float wyv = wy[y];
        for (int x = 0; x < 512; x++) {
            int xa = x0[x], xb = x1[x]; float wxv = wx[x];
            for (int c = 0; c < 3; c++) {
                float a = src[(ya * 1280 + xa) * 3 + c];
                float b = src[(ya * 1280 + xb) * 3 + c];
                float d = src[(yb * 1280 + xa) * 3 + c];
                float e = src[(yb * 1280 + xb) * 3 + c];
                float top = a + (b - a) * wxv;
                float bot = d + (e - d) * wxv;
                float val = top + (bot - top) * wyv;
                int out = (int)(val + 0.5f);
                if (out < 0) out = 0; if (out > 255) out = 255;
                int ch = c == 0 ? 2 : (c == 1 ? 1 : 0);
                dst[(y * 512 + x) * 3 + ch] = (unsigned char)out;
            }
        }
    }
}

static float qval(const void *buf, int is_float, Attr *a, uint32_t idx) {
    if (is_float) return ((float *)buf)[idx];
    int8_t v = ((int8_t *)buf)[idx];
    return (v - a->zp) * a->scale;
}

static int cmp_det(const void *a, const void *b) {
    float x = ((Det *)a)->score, y = ((Det *)b)->score;
    return x < y ? 1 : (x > y ? -1 : 0);
}

static float iou(Det a, Det b) {
    float x1 = fmaxf(a.x1, b.x1), y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2), y2 = fminf(a.y2, b.y2);
    float inter = fmaxf(0, x2 - x1) * fmaxf(0, y2 - y1);
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / fmaxf(aa + bb - inter, 1e-9f);
}

static int decode_det(void **bufs, Attr *attrs, int is_float, Det *out) {
    Det cand[MAX_DET]; int nc = 0;
    for (int s = 0; s < 4; s++) {
        Attr *ra = &attrs[s * 3], *ca = &attrs[s * 3 + 1];
        int C = ra->dims[1], H = ra->dims[2], W = ra->dims[3];
        (void)C;
        float sx = 512.0f / W, sy = 288.0f / H;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = y * W + x;
                int best = 0; float bs = -1;
                for (int c = 0; c < 10; c++) {
                    float v = qval(bufs[s * 3 + 1], is_float, ca, (uint32_t)c * H * W + idx);
                    if (v > bs) { bs = v; best = c; }
                }
                if (bs < CONF) continue;
                float d[4];
                for (int side = 0; side < 4; side++) {
                    float m = -1e30f;
                    for (int b = 0; b < 16; b++) {
                        float v = qval(bufs[s * 3], is_float, ra, (uint32_t)(side * 16 + b) * H * W + idx);
                        if (v > m) m = v;
                    }
                    float sum = 0, exp_sum = 0;
                    for (int b = 0; b < 16; b++) {
                        float v = qval(bufs[s * 3], is_float, ra, (uint32_t)(side * 16 + b) * H * W + idx);
                        float e = expf(v - m); exp_sum += e; sum += b * e;
                    }
                    d[side] = sum / exp_sum;
                }
                float cx = (x + 0.5f - d[0]) * sx;
                float cy = (y + 0.5f - d[1]) * sy;
                float rx = (x + 0.5f + d[2]) * sx;
                float by = (y + 0.5f + d[3]) * sy;
                if (rx <= cx || by <= cy) continue;
                if (nc >= MAX_DET) continue;
                cand[nc++] = (Det){cx * 2.5f, cy * 2.5f, rx * 2.5f, by * 2.5f, bs, best};
            }
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
                Det *k = &out[n - kept + j];
                if (iou(tmp[i], *k) > NMS_IOU) { ok = 0; break; }
            }
            if (ok && n < MAX_DET) out[n++] = tmp[i], kept++;
        }
    }
    return n;
}

static void seg_stats(void *bufs[15], Attr *attrs, int is_float, int *lane_fg, int *cross,
                      int *solid, int *dash, int *direct, int *alt, uint8_t *lane_mask,
                      uint8_t *style_mask, uint8_t *drv_mask) {
    Attr *la = &attrs[12], *sa = &attrs[13], *da = &attrs[14];
    int H = la->dims[2], W = la->dims[3], HW = H * W;
    if (is_float) {
        const float *lp = (const float *)bufs[12];
        const float *sp = (const float *)bufs[13];
        const float *dp = (const float *)bufs[14];
        for (int i = 0; i < HW; i++) {
            int best = 0; float bv = lp[i];
            for (int c = 1; c < 9; c++) { float v = lp[c * HW + i]; if (v > bv) { bv = v; best = c; } }
            lane_mask[i] = (uint8_t)best;
            if (best != 8) (*lane_fg)++;
            if (best == 0) (*cross)++;
            int sb = 0; float sv = sp[i];
            for (int c = 1; c < 3; c++) { float v = sp[c * HW + i]; if (v > sv) { sv = v; sb = c; } }
            style_mask[i] = (uint8_t)sb;
            if (best != 8 && sb == 0) (*solid)++;
            if (best != 8 && sb == 1) (*dash)++;
            int db = 0; float dv = dp[i];
            for (int c = 1; c < 3; c++) { float v = dp[c * HW + i]; if (v > dv) { dv = v; db = c; } }
            drv_mask[i] = (uint8_t)db;
            if (db == 0) (*direct)++;
            if (db == 1) (*alt)++;
        }
    } else {
        const int8_t *lp = (const int8_t *)bufs[12];
        const int8_t *sp = (const int8_t *)bufs[13];
        const int8_t *dp = (const int8_t *)bufs[14];
        for (int i = 0; i < HW; i++) {
            int best = 0; int8_t bv = lp[i];
            for (int c = 1; c < 9; c++) { int8_t v = lp[c * HW + i]; if (v > bv) { bv = v; best = c; } }
            lane_mask[i] = (uint8_t)best;
            if (best != 8) (*lane_fg)++;
            if (best == 0) (*cross)++;
            int sb = 0; int8_t sv = sp[i];
            for (int c = 1; c < 3; c++) { int8_t v = sp[c * HW + i]; if (v > sv) { sv = v; sb = c; } }
            style_mask[i] = (uint8_t)sb;
            if (best != 8 && sb == 0) (*solid)++;
            if (best != 8 && sb == 1) (*dash)++;
            int db = 0; int8_t dv = dp[i];
            for (int c = 1; c < 3; c++) { int8_t v = dp[c * HW + i]; if (v > dv) { dv = v; db = c; } }
            drv_mask[i] = (uint8_t)db;
            if (db == 0) (*direct)++;
            if (db == 1) (*alt)++;
        }
    }
}

static void stats(const double *v, int n, double *mean, double *p50, double *p95) {
    double *a = malloc(sizeof(double) * n);
    memcpy(a, v, sizeof(double) * n);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (a[j] < a[i]) { double t = a[i]; a[i] = a[j]; a[j] = t; }
    double s = 0; for (int i = 0; i < n; i++) s += a[i];
    *mean = s / n; *p50 = a[n / 2]; *p95 = a[(int)(n * 0.95) >= n ? n - 1 : (int)(n * 0.95)];
    free(a);
}

static void get_masks(rknn_context ctx, rknn_input_output_num io, Attr *attrs, int is_float,
                      unsigned char *rgb, uint8_t *lane, uint8_t *style, uint8_t *drv) {
    rknn_input in; memset(&in, 0, sizeof(in));
    in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    in.size = 512 * 288 * 3; in.buf = rgb;
    rknn_inputs_set(ctx, 1, &in);
    rknn_output outs[MAX_OUT]; memset(outs, 0, sizeof(outs));
    for (uint32_t i = 0; i < io.n_output; i++) { outs[i].index = i; outs[i].want_float = is_float; }
    rknn_run(ctx, NULL);
    rknn_outputs_get(ctx, io.n_output, outs, NULL);
    void *bufs[MAX_OUT];
    for (uint32_t i = 0; i < io.n_output; i++) bufs[i] = outs[i].buf;
    int lf = 0, cw = 0, so = 0, da = 0, di = 0, al = 0;
    seg_stats(bufs, attrs, is_float, &lf, &cw, &so, &da, &di, &al, lane, style, drv);
    rknn_outputs_release(ctx, io.n_output, outs);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.rknn result.json [mode: -1=both,1=float,0=int8]\n", argv[0]); return 1; }
    int only_mode = argc > 3 ? atoi(argv[3]) : -1;
    size_t msize, fsize;
    unsigned char *model = read_file(argv[1], &msize);
    unsigned char *raw = read_file("/home/cat/b05g_113421_3s.rgb", &fsize);
    init_resize();
    rknn_context ctx;
    if (rknn_init(&ctx, model, msize, 0, NULL)) { fprintf(stderr, "init failed\n"); return 1; }
    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    Attr attrs[MAX_OUT];
    for (uint32_t i = 0; i < io.n_output; i++) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        attrs[i] = (Attr){a.n_elems, {a.dims[0], a.dims[1], a.dims[2], a.dims[3]},
                          a.scale, a.zp, a.type, a.fmt, ""};
        snprintf(attrs[i].name, sizeof(attrs[i].name), "%s", a.name);
        printf("ATTR %u %s type=%d fmt=%d qnt=%d zp=%d scale=%.8f dims=%ux%ux%ux%u\n",
               i, a.name, a.type, a.fmt, a.qnt_type, a.zp, a.scale,
               a.dims[0], a.dims[1], a.dims[2], a.dims[3]);
    }
    int is_mtl = io.n_output >= 15;
    int seg_hw = is_mtl ? (int)(attrs[12].dims[2] * attrs[12].dims[3]) : 0;
    unsigned char *bgr = malloc(1280 * 720 * 3);
    unsigned char *rgb = malloc(512 * 288 * 3);
    Det dets[MAX_DET];
    uint8_t *lane = malloc(288 * 512), *style = malloc(288 * 512), *drv = malloc(288 * 512);

    long long g_lane = 0, g_cross = 0, g_solid = 0, g_dash = 0, g_direct = 0, g_alt = 0;
    for (int mode = 1; mode >= 0; mode--) {
        if (only_mode >= 0 && mode != only_mode) continue;
        long long seg_lane = 0, seg_cross = 0, seg_solid = 0, seg_dash = 0, seg_direct = 0, seg_alt = 0;
        int nframes = 300;
        double *pre = malloc(sizeof(double) * nframes), *run = malloc(sizeof(double) * nframes);
        double *get = malloc(sizeof(double) * nframes), *det_t = malloc(sizeof(double) * nframes);
        double *seg_t = malloc(sizeof(double) * nframes), *rel = malloc(sizeof(double) * nframes);
        double *tot = malloc(sizeof(double) * nframes), *perf = malloc(sizeof(double) * nframes);
        int k = 0, det_count = 0;
        double tstart = now_ms();
        for (int round = 0; round < 5; round++) {
            for (int f = 0; f < 60; f++) {
                double t0 = now_ms();
                decode565_rgb((uint16_t *)(raw + (size_t)f * 1280 * 720 * 2), bgr, 1280, 720);
                resize_bgr_to_rgb(bgr, rgb);
                double t1 = now_ms();
                rknn_input in; memset(&in, 0, sizeof(in));
                in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
                in.size = 512 * 288 * 3; in.buf = rgb;
                rknn_inputs_set(ctx, 1, &in);
                rknn_output outs[MAX_OUT]; memset(outs, 0, sizeof(outs));
                for (uint32_t i = 0; i < io.n_output; i++) { outs[i].index = i; outs[i].want_float = mode; }
                double t2 = now_ms();
                rknn_run(ctx, NULL);
                double t3 = now_ms();
                rknn_outputs_get(ctx, io.n_output, outs, NULL);
                double t4 = now_ms();
                rknn_perf_run pr; memset(&pr, 0, sizeof(pr));
                rknn_query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr));
                void *bufs[MAX_OUT];
                for (uint32_t i = 0; i < io.n_output; i++) bufs[i] = outs[i].buf;
                double t5 = now_ms();
                Det local[MAX_DET];
                int nd = decode_det(bufs, attrs, mode, local);
                double t6 = now_ms();
                if (is_mtl) {
                    int lf = 0, cw = 0, so = 0, da = 0, di = 0, al = 0;
                    seg_stats(bufs, attrs, mode, &lf, &cw, &so, &da, &di, &al, lane, style, drv);
                    seg_lane += lf; seg_cross += cw; seg_solid += so; seg_dash += da;
                    seg_direct += di; seg_alt += al;
                }
                double t7 = now_ms();
                rknn_outputs_release(ctx, io.n_output, outs);
                double t8 = now_ms();
                pre[k] = t1 - t0; run[k] = t3 - t2; get[k] = t4 - t3;
                det_t[k] = t6 - t5; seg_t[k] = t7 - t6; rel[k] = t8 - t7;
                tot[k] = t8 - t0; perf[k] = pr.run_duration / 1000.0;
                det_count += nd; k++;
            }
        }
        double wall = (now_ms() - tstart) / 1000.0;
        double m, p50, p95;
        printf("MODE %s outputs=%u infer=%d frames=%d det=%d wall=%.3f fps=%.2f\n",
               mode ? "float" : "int8", io.n_output, is_mtl, k, det_count, wall, k / wall);
        stats(pre, k, &m, &p50, &p95); printf("pre mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(run, k, &m, &p50, &p95); printf("run mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(get, k, &m, &p50, &p95); printf("get mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(det_t, k, &m, &p50, &p95); printf("det mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(seg_t, k, &m, &p50, &p95); printf("seg mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(rel, k, &m, &p50, &p95); printf("rel mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(tot, k, &m, &p50, &p95); printf("total mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        stats(perf, k, &m, &p50, &p95); printf("perf mean=%.2f p50=%.2f p95=%.2f\n", m, p50, p95);
        if (is_mtl) {
            double px = (double)k * seg_hw;
            printf("SEG %s lane=%.4f cross=%.4f solid_in_lane=%.4f dash_in_lane=%.4f direct=%.4f alt=%.4f\n",
                   mode ? "float" : "int8", seg_lane / px, seg_cross / px,
                   seg_solid / fmax((double)seg_lane, 1), seg_dash / fmax((double)seg_lane, 1),
                   seg_direct / px, seg_alt / px);
        }
        free(pre); free(run); free(get); free(det_t); free(seg_t); free(rel); free(tot); free(perf);
    }
    if (is_mtl) {
        uint8_t *fl = malloc(288 * 512), *fs = malloc(288 * 512), *fd = malloc(288 * 512);
        uint8_t *ql = malloc(288 * 512), *qs = malloc(288 * 512), *qd = malloc(288 * 512);
        long long lane_ok = 0, style_ok = 0, style_n = 0, drv_ok = 0, pixels = 0;
        for (int f = 0; f < 60; f++) {
            decode565_rgb((uint16_t *)(raw + (size_t)f * 1280 * 720 * 2), bgr, 1280, 720);
            resize_bgr_to_rgb(bgr, rgb);
            get_masks(ctx, io, attrs, 1, rgb, fl, fs, fd);
            get_masks(ctx, io, attrs, 0, rgb, ql, qs, qd);
            for (int i = 0; i < seg_hw; i++) {
                pixels++;
                if (fl[i] == ql[i]) lane_ok++;
                if (fl[i] != 8) { style_n++; if (fs[i] == qs[i]) style_ok++; }
                if (fd[i] == qd[i]) drv_ok++;
            }
        }
        printf("CONSISTENCY lane_agree=%.6f style_agree_in_lane=%.6f drv_agree=%.6f pixels=%lld\n",
               (double)lane_ok / pixels, (double)style_ok / fmax((double)style_n, 1),
               (double)drv_ok / pixels, pixels);
        free(fl); free(fs); free(fd); free(ql); free(qs); free(qd);
    }
    rknn_destroy(ctx);
    free(model); free(raw); free(bgr); free(rgb); free(lane); free(style); free(drv);
    return 0;
}
