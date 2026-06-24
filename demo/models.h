/* Switchable model registry for the demo. The demo can load several characters and cycle between
 * them at runtime (the M key), so a model can be picked at runtime rather than only via a
 * launch-config `--gltf` argument. Item 0 is always the procedural biped; the rest are discovered by scanning the demo's
 * assets folder for `.glb`/`.gltf` files that have a sibling baked `.mrw` (the offline pipeline's
 * output, e.g. the `demo_ual1` target). Drop a baked model into that folder and it appears in the
 * cycle. This header is demo-only and touches no marrow runtime types. */
#ifndef DEMO_MODELS_H
#define DEMO_MODELS_H

#include <stddef.h>

typedef enum { MODEL_PROCEDURAL = 0, MODEL_GLTF } ModelKind;

typedef struct {
    ModelKind kind;
    char name[64];     /* display label (the file stem, or "procedural biped") */
    char gltf[1024];   /* MODEL_GLTF: source .glb/.gltf path                   */
    char mrw[1024];    /* MODEL_GLTF: sibling baked .mrw the demo loads         */
} ModelSource;

#define MODELS_MAX 64

typedef struct {
    ModelSource items[MODELS_MAX];
    int count;
} ModelRegistry;

/* Fill `m` as the always-available procedural biped. */
void model_source_procedural(ModelSource *m);
/* Fill `m` as a glTF model. `mrw` may be NULL to derive the sibling `<stem>.mrw` from `gltf`. */
void model_source_from_gltf(ModelSource *m, const char *gltf, const char *mrw);

/* Populate `reg`: item 0 = procedural biped, then every baked model found next to the executable
 * (`<exe_dir>/../assets`, else `./assets`), then `explicit` (if non-NULL and not already present).
 * Returns the index that should be the startup model - `explicit`'s index if given, else 0. */
int models_discover(ModelRegistry *reg, const ModelSource *explicit_model);

#endif /* DEMO_MODELS_H */
