/* Emits the committed golden .mrw fixture. Run once and commit the output:
 *   gen_fixtures <path-to-fixtures-dir>   (default: ./)  */
#include "fixture_rig.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[1024];
    snprintf(path, sizeof path, "%s/rig.mrw", dir);

    uint8_t *buf = NULL;
    size_t sz = build_fixture(&buf);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); mrw_free(buf); return 1; }
    size_t wrote = fwrite(buf, 1, sz, f);
    fclose(f);
    mrw_free(buf);
    if (wrote != sz) { fprintf(stderr, "short write\n"); return 1; }
    printf("wrote %s (%zu bytes)\n", path, sz);
    return 0;
}
