/* Tiny persistent thread pool - the demo's job scheduler for multithreaded marrow.
 *
 * marrow's runtime owns no threads: every batch/pose call is a side-effect-free function over
 * caller-owned buffers, so the *application* schedules the work. This is the schedule: a fixed
 * worker pool plus a blocking parallel-for, the same shape an engine would use to fan sampling,
 * blend, and transform jobs across cores. The marrow runtime never sees it.
 *
 * Contract that makes the fan-out safe (and is exactly what makes marrow jobifiable):
 *   - read-only inputs (skeleton/clip/baked views, the .mrw blob, the mrw_dispatch) are SHARED;
 *   - each worker writes a DISJOINT output slice and uses its OWN scratch (never shared);
 *   - jobs_parallel_for is a barrier: all chunks complete before it returns.
 *
 * Portable C11: Win32 (CONDITION_VARIABLE) and POSIX (pthreads). Demo-only - not in the runtime. */
#ifndef DEMO_JOBS_H
#define DEMO_JOBS_H

#include <stdint.h>

#define JOBS_MAX_WORKERS 64u

typedef struct Jobs Jobs;

/* Called once per worker with that worker's half-open item range [begin, end). `worker` in
 * [0, worker_count) selects per-worker resources (e.g. a private scratch slice). */
typedef void (*JobFn)(void *ctx, uint32_t worker, uint32_t begin, uint32_t end);

/* Create a pool of `worker_count` lanes of parallelism (the calling thread is lane 0, so
 * worker_count-1 background threads are spawned). 0 ⇒ auto = host hardware threads. Clamped to
 * [1, JOBS_MAX_WORKERS]. If some background threads fail to start, the pool degrades to as many
 * lanes as did start (≥1) rather than failing - query the actual count with jobs_worker_count.
 * Returns NULL only if the pool struct itself can't be allocated. */
Jobs    *jobs_create(uint32_t worker_count);
uint32_t jobs_worker_count(const Jobs *j);   /* actual lanes (may be < requested); 1 if j==NULL */

/* Split [0, item_count) into worker_count contiguous, balanced chunks and run `fn` on each - chunk 0
 * on the calling thread, the rest on the pool - then block until all complete. item_count==0 is a
 * no-op. Chunk boundaries are arbitrary (not aligned to any internal tile), which is the point: the
 * callback owns honoring per-call homogeneity (e.g. clip grouping).
 *
 * NOT REENTRANT: a single dispatcher thread owns the pool. Do not call jobs_parallel_for again
 * (from another thread, or from inside a job callback) while one is in flight on the same pool. */
void jobs_parallel_for(Jobs *j, uint32_t item_count, JobFn fn, void *ctx);

/* Destroy the pool. Precondition: no jobs_parallel_for is in flight (the pool is idle) - call from
 * the same dispatcher thread after the last parallel_for has returned. */
void jobs_destroy(Jobs *j);

#endif /* DEMO_JOBS_H */
