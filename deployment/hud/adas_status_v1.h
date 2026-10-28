/* ADST wire proposal: explicit little-endian bytes, never native struct memcpy.
 * No BAR/ioctl allocation is implied. FPGA clock-domain agreement is required.
 */
#ifndef ADAS_EDGE_STATUS_V1_H
#define ADAS_EDGE_STATUS_V1_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#define ADST_V1_SIZE 60u
enum adst_light { ADST_LIGHT_UNKNOWN=0, ADST_RED=1, ADST_YELLOW=2, ADST_GREEN=3 };
enum adst_lane { ADST_LANE_UNKNOWN=0, ADST_SOLID=1, ADST_DASHED=2 };
enum adst_action {
    ADST_ACTION_UNKNOWN=0, ADST_STOP=1, ADST_FORWARD=2,
    ADST_LEFT=3, ADST_RIGHT=4, ADST_BRAKE=5, ADST_WARNING=6
};
enum adst_clock { ADST_VIDEO_REPLAY=1, ADST_HOST_MONOTONIC=2, ADST_FPGA_SOURCE_NS=3 };
enum adst_valid {
    ADST_SIGNAL_OBSERVED=1, ADST_SIGNAL_ASSOCIATED=2,
    ADST_LEFT_LANE=4, ADST_RIGHT_LANE=8, ADST_DECISION=16
};
enum adst_warning {
    ADST_RED_LIGHT=1, ADST_SOLID_LANE=2, ADST_PEDESTRIAN=4, ADST_COLLISION=8,
    ADST_UNKNOWN_SIGNAL=16, ADST_UNKNOWN_LANE=32, ADST_STALE=64,
    ADST_OVERSPEED=128, ADST_UNVERIFIED_ASSOCIATION=256, ADST_PROTOCOL_ERROR=512
};

struct adst_v1 {
    uint32_t sequence, stream_id;
    uint64_t source_frame_id, source_timestamp_ns;
    uint8_t light, left_lane, right_lane, action;
    uint16_t confidence_q15, clock_domain;
    uint32_t valid, warnings, valid_for_ms;
};

static inline void adst_put(uint8_t *p, uint64_t value, unsigned count)
{
    unsigned i;
    for (i=0; i<count; ++i) p[i]=(uint8_t)(value >> (8*i));
}
static inline uint64_t adst_get(const uint8_t *p, unsigned count)
{
    uint64_t value=0;
    unsigned i;
    for (i=0; i<count; ++i) value |= (uint64_t)p[i] << (8*i);
    return value;
}
static inline uint32_t adst_crc32(const uint8_t *p, size_t count)
{
    uint32_t crc=UINT32_C(0xffffffff);
    size_t i;
    unsigned bit;
    for (i=0; i<count; ++i) {
        crc ^= p[i];
        for (bit=0; bit<8; ++bit)
            crc=(crc >> 1) ^ (UINT32_C(0xedb88320) & (uint32_t)-(int32_t)(crc & 1));
    }
    return ~crc;
}
static inline int adst_v1_valid(const struct adst_v1 *s)
{
    if (!s || !s->sequence || s->light > 3 || s->left_lane > 2 || s->right_lane > 2
        || s->action > 6 || s->confidence_q15 > 32767 || s->clock_domain < 1
        || s->clock_domain > 3 || (s->valid & ~UINT32_C(31))
        || (s->warnings & ~UINT32_C(1023)) || !s->valid_for_ms || s->valid_for_ms > 60000)
        return -EINVAL;
    if ((s->valid & 2) && !(s->valid & 1)) return -EINVAL;
    if ((s->valid & 1) && !s->light) return -EINVAL;
    if ((s->valid & 4) && !s->left_lane) return -EINVAL;
    if ((s->valid & 8) && !s->right_lane) return -EINVAL;
    if ((s->valid & 16) && !s->action) return -EINVAL;
    return 0;
}
static inline int adst_v1_encode(const struct adst_v1 *s, uint8_t p[ADST_V1_SIZE])
{
    if (!p || adst_v1_valid(s)) return -EINVAL;
    memset(p, 0, ADST_V1_SIZE);
    memcpy(p, "ADST", 4);
    adst_put(p+4, 1, 2); adst_put(p+6, ADST_V1_SIZE, 2);
    adst_put(p+8, s->sequence, 4); adst_put(p+12, s->stream_id, 4);
    adst_put(p+16, s->source_frame_id, 8); adst_put(p+24, s->source_timestamp_ns, 8);
    p[32]=s->light; p[33]=s->left_lane; p[34]=s->right_lane; p[35]=s->action;
    adst_put(p+36, s->confidence_q15, 2); adst_put(p+38, s->clock_domain, 2);
    adst_put(p+40, s->valid, 4); adst_put(p+44, s->warnings, 4);
    adst_put(p+48, s->valid_for_ms, 4); adst_put(p+56, adst_crc32(p,56), 4);
    return 0;
}
static inline int adst_v1_decode(const uint8_t *p, size_t count, struct adst_v1 *s)
{
    struct adst_v1 decoded;
    if (!p || !s || count != ADST_V1_SIZE || memcmp(p,"ADST",4)
        || adst_get(p+4,2) != 1 || adst_get(p+6,2) != ADST_V1_SIZE
        || adst_get(p+52,4) || adst_get(p+56,4) != adst_crc32(p,56)) return -EINVAL;
    decoded.sequence=(uint32_t)adst_get(p+8,4); decoded.stream_id=(uint32_t)adst_get(p+12,4);
    decoded.source_frame_id=adst_get(p+16,8); decoded.source_timestamp_ns=adst_get(p+24,8);
    decoded.light=p[32]; decoded.left_lane=p[33]; decoded.right_lane=p[34]; decoded.action=p[35];
    decoded.confidence_q15=(uint16_t)adst_get(p+36,2); decoded.clock_domain=(uint16_t)adst_get(p+38,2);
    decoded.valid=(uint32_t)adst_get(p+40,4); decoded.warnings=(uint32_t)adst_get(p+44,4);
    decoded.valid_for_ms=(uint32_t)adst_get(p+48,4);
    if (adst_v1_valid(&decoded)) return -EINVAL;
    *s=decoded;
    return 0;
}
#endif
