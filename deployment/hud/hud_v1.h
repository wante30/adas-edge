/* Portable userspace codec. The existing Linux driver supplies MMIO access. */
#ifndef ADAS_EDGE_HUD_V1_H
#define ADAS_EDGE_HUD_V1_H
#include <stdint.h>
#include <errno.h>

#define HUD_V1_MAGIC UINT32_C(0x31445548)
#define HUD_V1_VERSION UINT32_C(0x00010000)
#define HUD_V1_CAPS UINT32_C(0x3f)
enum hud_v1_register {
    HUD_V1_MAGIC_OFFSET=0x600, HUD_V1_VERSION_OFFSET=0x604,
    HUD_V1_REQUEST_OFFSET=0x608, HUD_V1_STATUS_OFFSET=0x60c,
    HUD_V1_ACCEPTED_SEQ_OFFSET=0x610, HUD_V1_APPLIED_SEQ_OFFSET=0x614,
    HUD_V1_RK_STATE_OFFSET=0x618, HUD_V1_EFFECTIVE_STATE_OFFSET=0x61c,
    HUD_V1_REJECT_COUNT_OFFSET=0x620, HUD_V1_LAST_ERROR_OFFSET=0x624,
    HUD_V1_REJECTED_SEQ_OFFSET=0x628, HUD_V1_SNAPSHOT_ID_OFFSET=0x62c,
    HUD_V1_CAPS_OFFSET=0x630, HUD_V1_LAST_ACCEPTED_REQUEST_OFFSET=0x634
};
enum hud_v1_status {
    HUD_V1_BUSY=1, HUD_V1_RK_OWNER=2, HUD_V1_DEHAZE_OVERRIDE=4,
    HUD_V1_ACCEPTED_VALID=8, HUD_V1_APPLIED_VALID=16, HUD_V1_REJECT_VALID=32,
    HUD_V1_EFFECTIVE_VALID=64, HUD_V1_CONTROL_READY=128
};
enum hud_v1_error {
    HUD_V1_NONE=0, HUD_V1_ERR_BUSY=1, HUD_V1_BAD_ACCESS=2,
    HUD_V1_RESERVED_BITS=3, HUD_V1_BAD_COMBINATION=4, HUD_V1_ZERO_SEQUENCE=5,
    HUD_V1_SEQUENCE_CONFLICT=6, HUD_V1_READ_ONLY_WRITE=7, HUD_V1_NOT_READY=8
};
#define HUD_V1_VISIBLE UINT32_C(0x001)
#define HUD_V1_LEFT UINT32_C(0x002)
#define HUD_V1_RIGHT UINT32_C(0x004)
#define HUD_V1_HAZARD UINT32_C(0x008)
#define HUD_V1_LOW_BEAM UINT32_C(0x010)
#define HUD_V1_HIGH_BEAM UINT32_C(0x020)
#define HUD_V1_FRONT_FOG UINT32_C(0x040)
#define HUD_V1_POSITION UINT32_C(0x080)
#define HUD_V1_ADAS_VISIBLE UINT32_C(0x100)

static inline int hud_v1_encode(uint32_t sequence, unsigned owner,
                               uint32_t state, unsigned blink, uint32_t *word)
{
    if (!word || !sequence || sequence > 65535 || owner > 1 || state > 0x1ff || blink > 7)
        return -EINVAL;
    if (!owner && (state || blink)) return -EINVAL;
    if ((blink & 1) && !(state & 2)) return -EINVAL;
    if ((blink & 2) && !(state & 4)) return -EINVAL;
    if ((blink & 4) && !(state & 8)) return -EINVAL;
    *word = (sequence << 16) | ((uint32_t)owner << 12) | ((uint32_t)blink << 9) | state;
    return 0;
}
#endif
