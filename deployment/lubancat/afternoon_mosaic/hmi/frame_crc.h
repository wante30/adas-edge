#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <zlib.h>
#if defined(__linux__) && defined(__aarch64__) && defined(__GNUC__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_acle.h>
// Same bounded IEEE-CRC fast path as the deployed MIT frame_validator.c.
static __attribute__((target("+crc"))) uint32_t adasArmCrc(const unsigned char *data, size_t length) {
    uint32_t crc = UINT32_MAX;
    while (length >= 8) {
        uint64_t word; std::memcpy(&word, data, sizeof word);
        crc = __crc32d(crc, word); data += 8; length -= 8;
    }
    while (length--) crc = __crc32b(crc, *data++);
    return crc ^ UINT32_MAX;
}
#endif
inline uint32_t adasFrameCrc(const unsigned char *data, size_t length) {
#if defined(__linux__) && defined(__aarch64__) && defined(__GNUC__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    static const bool accelerated = (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
    if (accelerated) return adasArmCrc(data, length);
#endif
    return static_cast<uint32_t>(::crc32(::crc32(0L, Z_NULL, 0), data, static_cast<uInt>(length)));
}
