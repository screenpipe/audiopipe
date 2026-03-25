/*
 * win_compat.h - POSIX compatibility layer for Windows (MSVC)
 *
 * Provides: mmap/munmap, opendir/readdir/closedir, pthread basics,
 *           gettimeofday, strdup, sysconf, open/fstat/close.
 *
 * Uses SRWLock (not CRITICAL_SECTION) so mutexes can be
 * statically initialized with PTHREAD_MUTEX_INITIALIZER = {0}.
 */
#ifndef WIN_COMPAT_H
#define WIN_COMPAT_H

#ifdef _WIN32

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========================================================================
 * GCC attributes (not supported by MSVC)
 * ======================================================================== */
#ifndef __attribute__
#define __attribute__(x)
#endif

/* ========================================================================
 * strdup
 * ======================================================================== */
#ifndef strdup
#define strdup _strdup
#endif

/* ========================================================================
 * open / close / fstat / read
 * ======================================================================== */
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif

static inline int compat_open(const char *path, int flags) {
    int fd;
    _sopen_s(&fd, path, flags | _O_BINARY, _SH_DENYNO, _S_IREAD);
    return fd;
}

#define open  compat_open
#define close _close
#define read  _read

struct compat_stat {
    int64_t st_size;
};

static inline int compat_fstat(int fd, struct compat_stat *buf) {
    struct _stat64 s;
    int r = _fstat64(fd, &s);
    if (r == 0) buf->st_size = s.st_size;
    return r;
}

#define fstat(fd, buf)  compat_fstat(fd, (struct compat_stat*)(buf))
#define stat            compat_stat

/* ========================================================================
 * mmap / munmap
 * ======================================================================== */
#define PROT_READ   0x1
#define MAP_PRIVATE 0x02
#define MAP_FAILED  ((void *)-1)

static inline void *compat_mmap(void *addr, size_t length, int prot,
                                int flags, int fd, long long offset) {
    (void)addr; (void)prot; (void)flags; (void)offset;
    HANDLE fh = (HANDLE)_get_osfhandle(fd);
    if (fh == INVALID_HANDLE_VALUE) return MAP_FAILED;

    HANDLE mapping = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) return MAP_FAILED;

    void *ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, length);
    CloseHandle(mapping);
    return ptr ? ptr : MAP_FAILED;
}

static inline int compat_munmap(void *addr, size_t length) {
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

#define mmap(a, l, p, f, fd, o)  compat_mmap(a, l, p, f, fd, o)
#define munmap(a, l)             compat_munmap(a, l)

/* ========================================================================
 * opendir / readdir / closedir
 * ======================================================================== */
struct dirent {
    char d_name[MAX_PATH];
};

typedef struct {
    HANDLE          handle;
    WIN32_FIND_DATAA data;
    struct dirent   ent;
    int             first;
} DIR;

static inline DIR *opendir(const char *path) {
    char pattern[MAX_PATH + 3];
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    DIR *d = (DIR *)calloc(1, sizeof(DIR));
    if (!d) return NULL;
    d->handle = FindFirstFileA(pattern, &d->data);
    if (d->handle == INVALID_HANDLE_VALUE) { free(d); return NULL; }
    d->first = 1;
    return d;
}

static inline struct dirent *readdir(DIR *d) {
    if (!d) return NULL;
    if (d->first) {
        d->first = 0;
    } else {
        if (!FindNextFileA(d->handle, &d->data)) return NULL;
    }
    strncpy(d->ent.d_name, d->data.cFileName, MAX_PATH - 1);
    d->ent.d_name[MAX_PATH - 1] = '\0';
    return &d->ent;
}

static inline int closedir(DIR *d) {
    if (!d) return -1;
    FindClose(d->handle);
    free(d);
    return 0;
}

/* ========================================================================
 * gettimeofday
 * ======================================================================== */
struct timeval {
    long tv_sec;
    long tv_usec;
};

static inline int gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    t -= 116444736000000000ULL; /* Jan 1 1601 -> Jan 1 1970 */
    t /= 10;                   /* 100ns -> us */
    tv->tv_sec  = (long)(t / 1000000ULL);
    tv->tv_usec = (long)(t % 1000000ULL);
    return 0;
}

/* ========================================================================
 * pthread (minimal: mutex via SRWLock, condvar, thread)
 *
 * SRWLock supports static init with SRWLOCK_INIT = {0}, unlike
 * CRITICAL_SECTION which requires runtime InitializeCriticalSection().
 * ======================================================================== */
typedef HANDLE pthread_t;
typedef int    pthread_attr_t;

typedef SRWLOCK            pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER  CONDITION_VARIABLE_INIT

static inline int pthread_mutex_init(pthread_mutex_t *m, const void *a) {
    (void)a; InitializeSRWLock(m); return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m) {
    (void)m; return 0; /* SRWLock has no destroy */
}
static inline int pthread_mutex_lock(pthread_mutex_t *m) {
    AcquireSRWLockExclusive(m); return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *m) {
    ReleaseSRWLockExclusive(m); return 0;
}

static inline int pthread_cond_init(pthread_cond_t *c, const void *a) {
    (void)a; InitializeConditionVariable(c); return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t *c) {
    (void)c; return 0;
}
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    SleepConditionVariableSRW(c, m, INFINITE, 0); return 0;
}
static inline int pthread_cond_signal(pthread_cond_t *c) {
    WakeConditionVariable(c); return 0;
}
static inline int pthread_cond_broadcast(pthread_cond_t *c) {
    WakeAllConditionVariable(c); return 0;
}

typedef struct {
    void *(*fn)(void *);
    void *arg;
} compat_thread_arg_t;

static inline unsigned __stdcall compat_thread_start(void *p) {
    compat_thread_arg_t *a = (compat_thread_arg_t *)p;
    a->fn(a->arg);
    free(a);
    return 0;
}

static inline int pthread_create(pthread_t *t, const pthread_attr_t *a,
                                 void *(*fn)(void *), void *arg) {
    (void)a;
    compat_thread_arg_t *ta = (compat_thread_arg_t *)malloc(sizeof(*ta));
    if (!ta) return -1;
    ta->fn = fn;
    ta->arg = arg;
    *t = (HANDLE)_beginthreadex(NULL, 0, compat_thread_start, ta, 0, NULL);
    return *t ? 0 : -1;
}

static inline int pthread_join(pthread_t t, void **retval) {
    (void)retval;
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    return 0;
}

/* ========================================================================
 * sysconf (_SC_NPROCESSORS_ONLN)
 * ======================================================================== */
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 1
#endif

static inline long sysconf(int name) {
    if (name == _SC_NPROCESSORS_ONLN) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwNumberOfProcessors;
    }
    return -1;
}

/* ========================================================================
 * Portable BESSEL_I0 (replaces GCC statement expression in qwen_asr_audio.c)
 * ======================================================================== */
static inline double compat_bessel_i0(double x) {
    double sum = 1.0, term = 1.0, xx = x * x;
    for (int k = 1; k <= 20; k++) {
        term *= xx / (4.0 * (double)k * (double)k);
        sum += term;
    }
    return sum;
}

#endif /* _WIN32 */
#endif /* WIN_COMPAT_H */
