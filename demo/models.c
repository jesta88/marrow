/* See models.h. The platform-specific bits - locating the executable and listing a directory - are
 * isolated here behind Win32 / POSIX branches so main.c stays portable. */

/* readlink/opendir/stat are POSIX names that glibc hides under strict -std=c11 (the demo target
 * sets C_EXTENSIONS OFF). The feature macro must be defined before ANY system header is pulled in,
 * so it sits above every #include. Ignored by MSVC. */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif

#include "models.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dirent.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static int ends_with_ci(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    if (n < m) return 0;
    for (size_t i = 0; i < m; ++i)
        if (tolower((unsigned char)s[n - m + i]) != tolower((unsigned char)suf[i])) return 0;
    return 1;
}

/* File stem (basename without directory or extension) into `out`. */
static void stem_of(const char *path, char *out, size_t cap) {
    const char *base = path;
    for (const char *p = path; *p; ++p) if (*p == '/' || *p == '\\') base = p + 1;
    size_t n = strlen(base);
    const char *dot = strrchr(base, '.');
    if (dot && dot > base) n = (size_t)(dot - base);
    if (n >= cap) n = cap - 1;
    memcpy(out, base, n);
    out[n] = 0;
}

/* Sibling baked-blob path for a glTF/glb: replace the extension with ".mrw". */
static void derive_mrw(const char *gltf, char *out, size_t cap) {
    size_t n = strlen(gltf);
    const char *dot = strrchr(gltf, '.');
    size_t stem = dot ? (size_t)(dot - gltf) : n;
    if (stem + 5 > cap) stem = cap > 5 ? cap - 5 : 0;   /* keep room for ".mrw\0" */
    memcpy(out, gltf, stem);
    memcpy(out + stem, ".mrw", 5);
}

void model_source_procedural(ModelSource *m) {
    memset(m, 0, sizeof *m);
    m->kind = MODEL_PROCEDURAL;
    snprintf(m->name, sizeof m->name, "procedural biped");
}

void model_source_from_gltf(ModelSource *m, const char *gltf, const char *mrw) {
    memset(m, 0, sizeof *m);
    m->kind = MODEL_GLTF;
    snprintf(m->gltf, sizeof m->gltf, "%s", gltf);
    if (mrw) snprintf(m->mrw, sizeof m->mrw, "%s", mrw);
    else     derive_mrw(gltf, m->mrw, sizeof m->mrw);
    stem_of(gltf, m->name, sizeof m->name);
}

/* Directory holding the executable, into `out` (no trailing slash). */
static int exe_dir(char *out, size_t cap) {
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, out, (DWORD)cap);
    if (n == 0 || n >= cap) return 0;
#else
    ssize_t n = readlink("/proc/self/exe", out, cap - 1);
    if (n <= 0) return 0;
    out[n] = 0;
#endif
    char *cut = NULL;
    for (char *p = out; *p; ++p) if (*p == '/' || *p == '\\') cut = p;
    if (!cut) return 0;
    *cut = 0;
    return 1;
}

static int dir_exists(const char *p) {
#ifdef _WIN32
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

/* The demo's assets dir. The offline pipeline writes the baked models to <demo-binary-dir>/assets,
 * which is the directory holding the executable, so "<exe_dir>/assets" is the primary location;
 * "<exe_dir>/../assets" and "./assets" are fallbacks for other run layouts. First match wins. */
static int assets_dir(char *out, size_t cap) {
    char ed[1024];
    if (exe_dir(ed, sizeof ed)) {
        snprintf(out, cap, "%s/assets", ed);
        if (dir_exists(out)) return 1;
        snprintf(out, cap, "%s/../assets", ed);
        if (dir_exists(out)) return 1;
    }
    snprintf(out, cap, "assets");
    return dir_exists(out);
}

static int find_by_name(const ModelRegistry *reg, const char *name) {
    for (int i = 0; i < reg->count; ++i)
        if (strcmp(reg->items[i].name, name) == 0) return i;
    return -1;
}

/* Append a glTF model unless one with the same display name is already present. */
static void add_model(ModelRegistry *reg, const char *gltf, const char *mrw) {
    if (reg->count >= MODELS_MAX) return;
    char nm[64];
    stem_of(gltf, nm, sizeof nm);
    if (find_by_name(reg, nm) >= 0) return;
    ModelSource *m = &reg->items[reg->count++];
    model_source_from_gltf(m, gltf, mrw);
}

static void scan_one(ModelRegistry *reg, const char *dir, const char *fname) {
    if (!ends_with_ci(fname, ".glb") && !ends_with_ci(fname, ".gltf")) return;
    char gltf[1024], mrw[1024];
    snprintf(gltf, sizeof gltf, "%s/%s", dir, fname);
    derive_mrw(gltf, mrw, sizeof mrw);
    if (!file_exists(mrw)) return;    /* only offer models that are actually baked */
    add_model(reg, gltf, mrw);
}

static void scan_dir(ModelRegistry *reg, const char *dir) {
#ifdef _WIN32
    char pat[1024];
    snprintf(pat, sizeof pat, "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) scan_one(reg, dir, fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) scan_one(reg, dir, e->d_name);
    closedir(d);
#endif
}

static int by_name(const void *a, const void *b) {
    return strcmp(((const ModelSource *)a)->name, ((const ModelSource *)b)->name);
}

int models_discover(ModelRegistry *reg, const ModelSource *explicit_model) {
    reg->count = 0;
    model_source_procedural(&reg->items[reg->count++]);

    char dir[1024];
    if (assets_dir(dir, sizeof dir)) {
        int before = reg->count;
        scan_dir(reg, dir);
        fprintf(stderr, "[models] assets dir '%s': %d baked model(s)\n", dir, reg->count - before);
    } else {
        fprintf(stderr, "[models] no assets dir found (looked next to the executable, then ./assets)\n");
    }

    /* Stable, alphabetical cycle order for the discovered models (dir listing order is unspecified);
     * the procedural biped stays at index 0. */
    if (reg->count > 2) qsort(&reg->items[1], (size_t)(reg->count - 1), sizeof reg->items[0], by_name);

    int startup = 0;
    if (explicit_model && explicit_model->kind == MODEL_GLTF) {
        int idx = find_by_name(reg, explicit_model->name);
        if (idx < 0 && reg->count < MODELS_MAX) {
            reg->items[reg->count] = *explicit_model;
            idx = reg->count++;
        }
        if (idx >= 0) startup = idx;
    }
    return startup;
}
