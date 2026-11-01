#include <cstddef>
#include <cstdint>

#include <rga/im2d.hpp>

extern "C" int adas_rga_bgr_resize_rgb(
    const uint8_t *source, int source_width, int source_height,
    int source_stride_bytes, uint8_t *destination, int destination_width,
    int destination_height, int destination_stride_bytes, int destination_x,
    int destination_y, int resized_width, int resized_height) {
    if (!source || !destination || source_width <= 0 || source_height <= 0 ||
        destination_width <= 0 || destination_height <= 0 ||
        source_stride_bytes < source_width * 3 ||
        destination_stride_bytes < destination_width * 3 ||
        source_stride_bytes % 3 != 0 || destination_stride_bytes % 3 != 0 ||
        destination_x < 0 || destination_y < 0 || resized_width <= 0 ||
        resized_height <= 0 || destination_x + resized_width > destination_width ||
        destination_y + resized_height > destination_height) {
        return IM_STATUS_INVALID_PARAM;
    }

    const rga_buffer_t src = wrapbuffer_virtualaddr_t(
        const_cast<uint8_t *>(source), source_width, source_height,
        source_stride_bytes / 3, source_height, RK_FORMAT_BGR_888);
    const rga_buffer_t dst = wrapbuffer_virtualaddr_t(
        destination, destination_width, destination_height,
        destination_stride_bytes / 3, destination_height, RK_FORMAT_RGB_888);
    const rga_buffer_t pattern{};
    const im_rect src_rect{0, 0, source_width, source_height};
    const im_rect dst_rect{destination_x, destination_y, resized_width, resized_height};
    const im_rect pattern_rect{};

    return static_cast<int>(improcess(src, dst, pattern, src_rect, dst_rect,
                                      pattern_rect, -1, nullptr, nullptr, IM_SYNC));
}
