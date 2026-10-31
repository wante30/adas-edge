#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model.rknn inputs.bin masks.bin\n", argv[0]); return 1; }
    FILE *mf = fopen(argv[1], "rb");
    if (!mf) { perror("model"); return 1; }
    fseek(mf, 0, SEEK_END); long ms = ftell(mf); fseek(mf, 0, SEEK_SET);
    unsigned char *model = malloc(ms);
    if (fread(model, 1, ms, mf) != (size_t)ms) { perror("model read"); return 1; }
    fclose(mf);
    FILE *inf = fopen(argv[2], "rb");
    if (!inf) { perror("inputs"); return 1; }
    fseek(inf, 0, SEEK_END); long is = ftell(inf); fseek(inf, 0, SEEK_SET);
    unsigned char *inputs = malloc(is);
    if (fread(inputs, 1, is, inf) != (size_t)is) { perror("inputs read"); return 1; }
    fclose(inf);
    if (is % (512 * 288 * 3) != 0) { fprintf(stderr, "bad inputs size %ld\n", is); return 1; }
    int frames = (int)(is / (512 * 288 * 3));

    rknn_context ctx;
    if (rknn_init(&ctx, model, ms, 0, NULL)) { fprintf(stderr, "init failed\n"); return 1; }
    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (io.n_output < 15) { fprintf(stderr, "expected 15 outputs\n"); return 1; }
    rknn_tensor_attr la, sa, da;
    memset(&la, 0, sizeof(la)); la.index = 12; rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &la, sizeof(la));
    memset(&sa, 0, sizeof(sa)); sa.index = 13; rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &sa, sizeof(sa));
    memset(&da, 0, sizeof(da)); da.index = 14; rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &da, sizeof(da));
    int H = la.dims[2], W = la.dims[3], HW = H * W;
    printf("seg shapes: lane %ux%u style %ux%u drv %ux%u frames=%d\n",
           la.dims[2], la.dims[3], sa.dims[2], sa.dims[3], da.dims[2], da.dims[3], frames);
    FILE *of = fopen(argv[3], "wb");
    if (!of) { perror("output"); return 1; }
    unsigned char *masks = malloc((size_t)HW * 3);
    for (int f = 0; f < frames; f++) {
        rknn_input in; memset(&in, 0, sizeof(in));
        in.index = 0; in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
        in.size = 512 * 288 * 3; in.buf = inputs + (size_t)f * 512 * 288 * 3;
        rknn_inputs_set(ctx, 1, &in);
        rknn_output outs[15]; memset(outs, 0, sizeof(outs));
        for (int i = 0; i < 15; i++) { outs[i].index = i; outs[i].want_float = 0; }
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, 15, outs, NULL);
        for (int h = 0; h < 3; h++) {
            const int8_t *p = (const int8_t *)outs[12 + h].buf;
            int C = (h == 0) ? 9 : 3;
            unsigned char *dst = masks + (size_t)h * HW;
            for (int i = 0; i < HW; i++) {
                int best = 0; int8_t bv = p[i];
                for (int c = 1; c < C; c++) {
                    int8_t v = p[(size_t)c * HW + i];
                    if (v > bv) { bv = v; best = c; }
                }
                dst[i] = (unsigned char)best;
            }
        }
        fwrite(masks, 1, (size_t)HW * 3, of);
        rknn_outputs_release(ctx, 15, outs);
    }
    fclose(of);
    rknn_destroy(ctx);
    free(model); free(inputs); free(masks);
    printf("wrote %s frames=%d\n", argv[3], frames);
    return 0;
}
