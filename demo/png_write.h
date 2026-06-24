/* Minimal, dependency-free PNG writer (8-bit RGBA) for demo screenshots.
 *
 * Uses uncompressed (stored) DEFLATE blocks, so there's no compressor - just CRC32 + Adler32.
 * Output is a valid PNG the Read tool can display; not size-optimized (screenshots only). */
#ifndef DEMO_PNG_WRITE_H
#define DEMO_PNG_WRITE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t png__crc(const uint8_t *p, size_t n, uint32_t crc) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
    }
    return ~crc;
}

static void png__put32(uint8_t *b, uint32_t v) { b[0]=(uint8_t)(v>>24); b[1]=(uint8_t)(v>>16); b[2]=(uint8_t)(v>>8); b[3]=(uint8_t)v; }

/* Write a chunk (length + type[4] + data + CRC). CRC covers type+data, so we hash them in one
 * pass over a temp buffer for simplicity (chunks here are small or one big IDAT - fine). */
static void png__chunk(FILE *f, const char *type, const uint8_t *data, uint32_t len) {
    uint8_t hdr[8];
    png__put32(hdr, len);
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (len) fwrite(data, 1, len, f);

    uint8_t *tmp = (uint8_t *)malloc((size_t)len + 4);
    memcpy(tmp, type, 4);
    if (len) memcpy(tmp + 4, data, len);
    uint8_t cb[4];
    png__put32(cb, png__crc(tmp, (size_t)len + 4, 0));
    free(tmp);
    fwrite(cb, 1, 4, f);
}

/* pixels: w*h RGBA8, top row first. Returns 0 on success. */
static int png_write_rgba(const char *path, uint32_t w, uint32_t h, const uint8_t *pixels) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    png__put32(ihdr, w); png__put32(ihdr + 4, h);
    ihdr[8] = 8;    /* bit depth   */
    ihdr[9] = 6;    /* RGBA        */
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    png__chunk(f, "IHDR", ihdr, 13);

    /* Raw filtered scanlines: filter byte 0 + RGBA per row. */
    size_t raw_len = (size_t)h * (1 + (size_t)w * 4);
    uint8_t *raw = (uint8_t *)malloc(raw_len);
    size_t o = 0;
    for (uint32_t y = 0; y < h; ++y) {
        raw[o++] = 0;
        memcpy(raw + o, pixels + (size_t)y * w * 4, (size_t)w * 4);
        o += (size_t)w * 4;
    }

    /* zlib stream: header + stored deflate blocks + adler32. */
    size_t nblocks = (raw_len + 65534) / 65535;
    if (nblocks == 0) nblocks = 1;
    size_t z_len = 2 + nblocks * 5 + raw_len + 4;
    uint8_t *z = (uint8_t *)malloc(z_len);
    size_t zo = 0;
    z[zo++] = 0x78; z[zo++] = 0x01;      /* zlib header (deflate, default) */
    size_t left = raw_len, ri = 0;
    while (left > 0 || (raw_len == 0 && ri == 0)) {
        size_t block = left > 65535 ? 65535 : left;
        uint8_t final = (left - block) == 0 ? 1 : 0;
        z[zo++] = final;                 /* BFINAL + BTYPE=00 */
        z[zo++] = (uint8_t)(block & 0xFF); z[zo++] = (uint8_t)(block >> 8);
        uint16_t nlen = (uint16_t)~block;
        z[zo++] = (uint8_t)(nlen & 0xFF); z[zo++] = (uint8_t)(nlen >> 8);
        memcpy(z + zo, raw + ri, block); zo += block;
        ri += block; left -= block;
        if (raw_len == 0) break;
    }
    /* adler32 over raw */
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw_len; ++i) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    uint32_t adler = (b << 16) | a;
    png__put32(z + zo, adler); zo += 4;

    png__chunk(f, "IDAT", z, (uint32_t)zo);
    png__chunk(f, "IEND", NULL, 0);

    free(z); free(raw);
    fclose(f);
    return 0;
}

#endif /* DEMO_PNG_WRITE_H */
