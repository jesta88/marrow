#include "jobs.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ platform threading shim */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <process.h>
typedef CRITICAL_SECTION   mtx_t_;
typedef CONDITION_VARIABLE  cnd_t_;
typedef HANDLE              thread_t_;
#  define MTX_INIT(m)      InitializeCriticalSection(m)
#  define MTX_DESTROY(m)   DeleteCriticalSection(m)
#  define MTX_LOCK(m)      EnterCriticalSection(m)
#  define MTX_UNLOCK(m)    LeaveCriticalSection(m)
#  define CND_INIT(c)      InitializeConditionVariable(c)
#  define CND_DESTROY(c)   ((void)(c))
#  define CND_WAIT(c, m)   SleepConditionVariableCS((c), (m), INFINITE)
#  define CND_SIGNAL(c)    WakeConditionVariable(c)
#  define CND_BROADCAST(c) WakeAllConditionVariable(c)
#  define THREAD_FN(name, arg) static unsigned __stdcall name(void *arg)
#  define THREAD_RET 0u
/* evaluates to 1 on success, 0 on failure */
#  define THREAD_START_OK(handle, fn, arg) (((handle) = (HANDLE)_beginthreadex(NULL, 0, (fn), (arg), 0, NULL)) != 0)
#  define THREAD_JOIN(handle) do { WaitForSingleObject((handle), INFINITE); CloseHandle(handle); } while (0)
static uint32_t hw_threads(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    return si.dwNumberOfProcessors ? (uint32_t)si.dwNumberOfProcessors : 1u;
}
#else
#  include <pthread.h>
#  include <unistd.h>
typedef pthread_mutex_t mtx_t_;
typedef pthread_cond_t  cnd_t_;
typedef pthread_t       thread_t_;
#  define MTX_INIT(m)      pthread_mutex_init((m), NULL)
#  define MTX_DESTROY(m)   pthread_mutex_destroy(m)
#  define MTX_LOCK(m)      pthread_mutex_lock(m)
#  define MTX_UNLOCK(m)    pthread_mutex_unlock(m)
#  define CND_INIT(c)      pthread_cond_init((c), NULL)
#  define CND_DESTROY(c)   pthread_cond_destroy(c)
#  define CND_WAIT(c, m)   pthread_cond_wait((c), (m))
#  define CND_SIGNAL(c)    pthread_cond_signal(c)
#  define CND_BROADCAST(c) pthread_cond_broadcast(c)
#  define THREAD_FN(name, arg) static void *name(void *arg)
#  define THREAD_RET NULL
/* evaluates to 1 on success, 0 on failure */
#  define THREAD_START_OK(handle, fn, arg) (pthread_create(&(handle), NULL, (fn), (arg)) == 0)
#  define THREAD_JOIN(handle) pthread_join((handle), NULL)
static uint32_t hw_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1u;
}
#endif

/* ------------------------------------------------------------------ pool */

typedef struct { Jobs *pool; uint32_t index; } Worker;

struct Jobs {
    uint32_t  worker_count;             /* lanes of parallelism, incl. the calling thread (lane 0) */
    uint32_t  nthreads;                 /* background threads = worker_count - 1                   */
    thread_t_ thread[JOBS_MAX_WORKERS]; /* [0, nthreads)                                           */
    Worker    warg[JOBS_MAX_WORKERS];   /* per-thread {pool,index} (index in [1, worker_count))    */

    mtx_t_ mtx;
    cnd_t_ cnd_work;  /* workers wait here for a new dispatch */
    cnd_t_ cnd_done;  /* the dispatcher waits here for completion */

    /* current job (published under mtx) */
    JobFn    fn;
    void    *ctx;
    uint32_t begin[JOBS_MAX_WORKERS], end[JOBS_MAX_WORKERS];
    uint64_t generation;  /* bumped per dispatch; workers run when it advances past their last-seen */
    uint32_t remaining;   /* background chunks still running this dispatch */
    int      shutdown;
};

/* Balanced contiguous split of [0,n) into `w` chunks (sizes differ by ≤1). */
static void split_chunks(uint32_t n, uint32_t w, uint32_t *begin, uint32_t *end) {
    uint32_t base = n / w, rem = n % w, off = 0;
    for (uint32_t i = 0; i < w; ++i) {
        uint32_t len = base + (i < rem ? 1u : 0u);
        begin[i] = off; end[i] = off + len; off += len;
    }
}

THREAD_FN(worker_main, arg) {
    Worker *self = (Worker *)arg;
    Jobs   *j    = self->pool;
    uint32_t idx = self->index;
    uint64_t seen = 0;

    MTX_LOCK(&j->mtx);
    for (;;) {
        while (!j->shutdown && j->generation == seen) CND_WAIT(&j->cnd_work, &j->mtx);
        if (j->shutdown) break;
        seen = j->generation;
        JobFn fn = j->fn; void *ctx = j->ctx;
        uint32_t b = j->begin[idx], e = j->end[idx];
        MTX_UNLOCK(&j->mtx);

        if (e > b) fn(ctx, idx, b, e);

        MTX_LOCK(&j->mtx);
        if (--j->remaining == 0) CND_SIGNAL(&j->cnd_done);
    }
    MTX_UNLOCK(&j->mtx);
    return THREAD_RET;
}

Jobs *jobs_create(uint32_t worker_count) {
    if (worker_count == 0) worker_count = hw_threads();
    if (worker_count < 1) worker_count = 1;
    if (worker_count > JOBS_MAX_WORKERS) worker_count = JOBS_MAX_WORKERS;

    Jobs *j = (Jobs *)calloc(1, sizeof *j);
    if (!j) return NULL;
    j->generation = 0;
    MTX_INIT(&j->mtx);
    CND_INIT(&j->cnd_work);
    CND_INIT(&j->cnd_done);

    /* Spawn worker_count-1 background lanes. If one fails to start, stop and degrade to the lanes
     * that did start - lane count is recomputed from the actual thread count, so callers (and
     * jobs_parallel_for's `remaining`) never wait on a thread that was never created. */
    uint32_t started = 0;
    for (uint32_t t = 0; t < worker_count - 1u; ++t) {
        j->warg[t].pool = j; j->warg[t].index = t + 1u;  /* background lanes are 1..worker_count-1 */
        if (!THREAD_START_OK(j->thread[t], worker_main, &j->warg[t])) break;
        started++;
    }
    j->nthreads = started;
    j->worker_count = started + 1u;   /* the dispatcher thread is lane 0 */
    return j;
}

uint32_t jobs_worker_count(const Jobs *j) { return j ? j->worker_count : 1u; }

void jobs_parallel_for(Jobs *j, uint32_t item_count, JobFn fn, void *ctx) {
    if (!j || item_count == 0) return;

    /* Single lane (or no pool threads): just run inline - no locking, no wakeups. */
    if (j->nthreads == 0) { fn(ctx, 0, 0, item_count); return; }

    MTX_LOCK(&j->mtx);
    j->fn = fn; j->ctx = ctx;
    split_chunks(item_count, j->worker_count, j->begin, j->end);
    j->remaining = j->nthreads;     /* the dispatcher runs lane 0 itself, below */
    j->generation++;
    CND_BROADCAST(&j->cnd_work);
    MTX_UNLOCK(&j->mtx);

    /* Lane 0 on the calling thread - full utilization, no idle dispatcher. */
    if (j->end[0] > j->begin[0]) fn(ctx, 0, j->begin[0], j->end[0]);

    MTX_LOCK(&j->mtx);
    while (j->remaining != 0) CND_WAIT(&j->cnd_done, &j->mtx);
    MTX_UNLOCK(&j->mtx);
}

void jobs_destroy(Jobs *j) {
    if (!j) return;
    MTX_LOCK(&j->mtx);
    j->shutdown = 1;
    CND_BROADCAST(&j->cnd_work);
    MTX_UNLOCK(&j->mtx);
    for (uint32_t t = 0; t < j->nthreads; ++t) THREAD_JOIN(j->thread[t]);
    CND_DESTROY(&j->cnd_done);
    CND_DESTROY(&j->cnd_work);
    MTX_DESTROY(&j->mtx);
    free(j);
}
