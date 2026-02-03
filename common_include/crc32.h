
/* 文件位置：common_include/crc32.h */

#ifdef __SSE4_2__
#include <nmmintrin.h>
static inline uint32_t calculate_crc32(const void* data, size_t length) {
    uint32_t crc = 0xffffffff;
    const uint8_t* p = (const uint8_t*)data;
    size_t i = 0;
    
    // 利用 SSE4.2 指令硬件加速：一次处理 8 字节
    for (; i + 8 <= length; i += 8) {
        crc = _mm_crc32_u64(crc, *(uint64_t*)(p + i));
    }
    // 处理剩余字节
    for (; i < length; i++) {
        crc = _mm_crc32_u8(crc, p[i]);
    }
    return crc ^ 0xffffffff;
}
#else

#ifndef CRC32_H
#define CRC32_H

#include <stdint.h>
#include <stddef.h>

static uint32_t crc32_table[256];
static int crc32_table_inited = 0;

// 运行时生成表，比硬编码几千个 hex 优雅且不易错
static void init_crc32_table(void) {
    uint32_t polynomial = 0xEDB88320;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            if (c & 1) {
                c = polynomial ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        crc32_table[i] = c;
    }
    crc32_table_inited = 1;
}

static inline uint32_t calculate_crc32(const void* data, size_t length) {
    if (!crc32_table_inited) init_crc32_table(); // 懒加载初始化
    
    uint32_t crc = 0xffffffff;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < length; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc & 0xff) ^ p[i]];
    }
    return crc ^ 0xffffffff;
}

#endif // CRC32_H
