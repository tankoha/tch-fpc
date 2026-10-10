/* Mock libtch for testing the generated Pascal layer without libtorch
   (cloud sessions, CI). Implements the handful of at_, ats_ and atg_ entry
   points that tests/mock/mock_test.pas exercises on a tiny CPU "tensor"
   (double data, up to 8 dims); every other atg_ function is a generated
   stub that reports "mock: not implemented" through the error slot
   (gen_stubs.sh). Built by tests/mock/run.sh into build/mock/libtch.so.
   Not a numerical reference: shapes, ownership and argument plumbing only. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "torch_api_generated.h"

typedef struct {
  int ndim;
  int64_t dims[8];
  int64_t numel;
  double *data;
  int kind;
} mock_tensor;

typedef struct {
  int is_float;
  int64_t i;
  double f;
} mock_scalar;

static __thread char *torch_last_err = NULL;
static int live_tensors = 0;
static int live_scalars = 0;

/* mock-only introspection for the Pascal test */
int mock_live_tensors(void) { return live_tensors; }
int mock_live_scalars(void) { return live_scalars; }
static int last_dim_null = -1, last_dim_len = -1;
int mock_last_dim_null(void) { return last_dim_null; }
int mock_last_dim_len(void) { return last_dim_len; }
static char last_mode[64];
static int last_value_null = -1;
static double last_value = 0;
const char *mock_last_mode(void) { return last_mode; }
int mock_last_value_null(void) { return last_value_null; }
double mock_last_value(void) { return last_value; }
static int64_t last_scalar_i = 0;
static double last_scalar_f = 0;
static int last_scalar_is_float = -1;
int mock_last_scalar_is_float(void) { return last_scalar_is_float; }
int64_t mock_last_scalar_i(void) { return last_scalar_i; }
double mock_last_scalar_f(void) { return last_scalar_f; }

static void set_err(const char *msg) {
  free(torch_last_err);
  torch_last_err = malloc(strlen(msg) + 160);
  sprintf(torch_last_err,
          "%s\nException raised from mock at mock_tch.c:0 (most recent call first):\n"
          "frame #0: mock_tch.so",
          msg);
}

void mock_unimpl(const char *name) {
  char buf[256];
  snprintf(buf, sizeof buf, "mock: not implemented: %s", name);
  set_err(buf);
}

char *get_and_reset_last_err(void) {
  char *e = torch_last_err;
  torch_last_err = NULL;
  return e;
}

void tchf_free(void *p) { free(p); }

static mock_tensor *mk(int ndim, const int64_t *dims, int kind) {
  mock_tensor *m = calloc(1, sizeof *m);
  m->ndim = ndim;
  m->numel = 1;
  for (int i = 0; i < ndim; i++) {
    m->dims[i] = dims[i];
    m->numel *= dims[i];
  }
  m->data = calloc(m->numel ? m->numel : 1, sizeof(double));
  m->kind = kind;
  live_tensors++;
  return m;
}

static mock_tensor *copy_of(const mock_tensor *s) {
  mock_tensor *m = mk(s->ndim, s->dims, s->kind);
  memcpy(m->data, s->data, s->numel * sizeof(double));
  return m;
}

static int same_shape(const mock_tensor *a, const mock_tensor *b) {
  if (a->ndim != b->ndim) return 0;
  for (int i = 0; i < a->ndim; i++)
    if (a->dims[i] != b->dims[i]) return 0;
  return 1;
}

/* hand-written torch_api.h subset */

void at_free(tensor t) {
  mock_tensor *m = t;
  if (!m) return;
  free(m->data);
  free(m);
  live_tensors--;
}
int at_defined(tensor t) { return t != NULL; }
size_t at_dim(tensor t) { return ((mock_tensor *)t)->ndim; }
void at_shape(tensor t, int64_t *dims) {
  mock_tensor *m = t;
  memcpy(dims, m->dims, m->ndim * sizeof(int64_t));
}
int at_scalar_type(tensor t) { return ((mock_tensor *)t)->kind; }
int at_device(tensor t) { (void)t; return -1; }

char *at_to_string(tensor t, int line_size) {
  mock_tensor *m = t;
  (void)line_size;
  char *s = malloc(32 * (m->numel + 2));
  char *p = s;
  p += sprintf(p, "[");
  for (int64_t i = 0; i < m->numel; i++)
    p += sprintf(p, "%s%g", i ? ", " : "", m->data[i]);
  sprintf(p, "]");
  return s;
}

static int64_t flat_index(const mock_tensor *m, const int64_t *idx, int n) {
  int64_t f = 0;
  for (int i = 0; i < n; i++) f = f * m->dims[i] + idx[i];
  return f;
}
double at_double_value_at_indexes(tensor t, int64_t *idx, int n) {
  mock_tensor *m = t;
  if (n != m->ndim) { set_err("mock: wrong number of indexes"); return 0; }
  return m->data[flat_index(m, idx, n)];
}
int64_t at_int64_value_at_indexes(tensor t, int64_t *idx, int n) {
  return (int64_t)at_double_value_at_indexes(t, idx, n);
}

scalar ats_int(int64_t v) {
  mock_scalar *s = malloc(sizeof *s);
  s->is_float = 0; s->i = v; s->f = 0;
  live_scalars++;
  return s;
}
scalar ats_float(double v) {
  mock_scalar *s = malloc(sizeof *s);
  s->is_float = 1; s->i = 0; s->f = v;
  live_scalars++;
  return s;
}
void ats_free(scalar s) {
  if (!s) return;
  free(s);
  live_scalars--;
}
static double scalar_value(scalar s) {
  mock_scalar *m = s;
  last_scalar_is_float = m->is_float;
  last_scalar_i = m->i;
  last_scalar_f = m->f;
  return m->is_float ? m->f : (double)m->i;
}

/* generated atg_* subset, same signatures as torch_api_generated.h */

void atg_ones(tensor *out__, int64_t *size_data, int size_len, int options_kind, int options_device) {
  (void)options_device;
  mock_tensor *m = mk(size_len, size_data, options_kind);
  for (int64_t i = 0; i < m->numel; i++) m->data[i] = 1.0;
  out__[0] = m;
}
void atg_zeros(tensor *out__, int64_t *size_data, int size_len, int options_kind, int options_device) {
  (void)options_device;
  out__[0] = mk(size_len, size_data, options_kind);
}

void atg_add(tensor *out__, tensor self, tensor other) {
  mock_tensor *a = self, *b = other;
  if (!same_shape(a, b)) { set_err("The size of tensor a must match the size of tensor b (mock)"); return; }
  mock_tensor *m = copy_of(a);
  for (int64_t i = 0; i < m->numel; i++) m->data[i] += b->data[i];
  out__[0] = m;
}
void atg_mul(tensor *out__, tensor self, tensor other) {
  mock_tensor *a = self, *b = other;
  if (!same_shape(a, b)) { set_err("The size of tensor a must match the size of tensor b (mock)"); return; }
  mock_tensor *m = copy_of(a);
  for (int64_t i = 0; i < m->numel; i++) m->data[i] *= b->data[i];
  out__[0] = m;
}
void atg_add_scalar(tensor *out__, tensor self, scalar other) {
  mock_tensor *m = copy_of(self);
  double v = scalar_value(other);
  for (int64_t i = 0; i < m->numel; i++) m->data[i] += v;
  out__[0] = m;
}

/* 1-D or 2-D, reduces along dim; keepdim keeps a size-1 dim */
void atg_max_dim(tensor *out__, tensor self, int64_t dim, int keepdim) {
  mock_tensor *s = self;
  if (s->ndim < 1 || s->ndim > 2 || dim < 0 || dim >= s->ndim) { set_err("mock: max_dim supports 1-D/2-D"); return; }
  int64_t rows = s->ndim == 2 ? s->dims[0] : 1, cols = s->ndim == 2 ? s->dims[1] : s->dims[0];
  int64_t outer = dim == 1 || s->ndim == 1 ? rows : cols;   /* number of results */
  int64_t odims[2]; int ondim;
  if (s->ndim == 1) { ondim = keepdim ? 1 : 0; odims[0] = 1; }
  else if (keepdim) { ondim = 2; odims[0] = dim == 0 ? 1 : rows; odims[1] = dim == 1 ? 1 : cols; }
  else { ondim = 1; odims[0] = outer; }
  mock_tensor *v = mk(ondim, odims, s->kind), *ix = mk(ondim, odims, 4);
  for (int64_t o = 0; o < outer; o++) {
    int64_t n = (s->ndim == 1) ? cols : (dim == 1 ? cols : rows);
    double best = 0; int64_t bi = 0;
    for (int64_t k = 0; k < n; k++) {
      double x = s->ndim == 1 ? s->data[k] : (dim == 1 ? s->data[o * cols + k] : s->data[k * cols + o]);
      if (k == 0 || x > best) { best = x; bi = k; }
    }
    v->data[o] = best; ix->data[o] = (double)bi;
  }
  out__[0] = v; out__[1] = ix;
}

tensor *atg_split(tensor self, int64_t split_size, int64_t dim) {
  mock_tensor *s = self;
  if (dim != 0 || s->ndim < 1 || split_size <= 0) { set_err("mock: split supports dim 0"); return NULL; }
  int64_t rowlen = s->numel / s->dims[0];
  int64_t n = (s->dims[0] + split_size - 1) / split_size;
  tensor *out__ = malloc((n + 1) * sizeof(tensor));
  for (int64_t c = 0; c < n; c++) {
    int64_t first = c * split_size, cnt = s->dims[0] - first < split_size ? s->dims[0] - first : split_size;
    int64_t dims[8]; memcpy(dims, s->dims, sizeof dims); dims[0] = cnt;
    mock_tensor *m = mk(s->ndim, dims, s->kind);
    memcpy(m->data, s->data + first * rowlen, cnt * rowlen * sizeof(double));
    out__[c] = m;
  }
  out__[n] = NULL;
  return out__;
}

int atg_is_nonzero(tensor self) {
  mock_tensor *s = self;
  if (s->numel != 1) { set_err("Boolean value of Tensor with more than one value is ambiguous (mock)"); return 0; }
  return s->data[0] != 0;
}

void atg_cat(tensor *out__, tensor *tensors_data, int tensors_len, int64_t dim) {
  if (tensors_len == 0) { set_err("torch.cat(): expected a non-empty list of Tensors (mock)"); return; }
  if (dim != 0) { set_err("mock: cat supports dim 0"); return; }
  mock_tensor *f = tensors_data[0];
  int64_t total = 0, rowlen = f->numel / f->dims[0];
  for (int i = 0; i < tensors_len; i++) {
    mock_tensor *t = tensors_data[i];
    if (t->ndim != f->ndim || t->numel / t->dims[0] != rowlen) { set_err("mock: cat shape mismatch"); return; }
    total += t->dims[0];
  }
  int64_t dims[8]; memcpy(dims, f->dims, sizeof dims); dims[0] = total;
  mock_tensor *m = mk(f->ndim, dims, f->kind);
  double *p = m->data;
  for (int i = 0; i < tensors_len; i++) {
    mock_tensor *t = tensors_data[i];
    memcpy(p, t->data, t->numel * sizeof(double));
    p += t->numel;
  }
  out__[0] = m;
}

/* 1-D: sorts self into the tensors passed in (values, indices), like out= */
void atg_sort_values(tensor *out__, tensor values, tensor indices, tensor self, int64_t dim, int descending) {
  mock_tensor *s = self, *v = values, *ix = indices;
  if (s->ndim != 1 || (dim != 0 && dim != -1)) { set_err("mock: sort_values supports 1-D"); return; }
  int64_t n = s->numel;
  free(v->data); free(ix->data);
  v->ndim = ix->ndim = 1; v->dims[0] = ix->dims[0] = n; v->numel = ix->numel = n;
  v->data = malloc(n * sizeof(double)); ix->data = malloc(n * sizeof(double));
  for (int64_t i = 0; i < n; i++) { v->data[i] = s->data[i]; ix->data[i] = (double)i; }
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = i + 1; j < n; j++)
      if (descending ? v->data[j] > v->data[i] : v->data[j] < v->data[i]) {
        double t = v->data[i]; v->data[i] = v->data[j]; v->data[j] = t;
        t = ix->data[i]; ix->data[i] = ix->data[j]; ix->data[j] = t;
      }
  out__[0] = copy_of(v);   /* libtorch returns handles to the same tensors */
  out__[1] = copy_of(ix);
}

/* records how the IntListOption arrived; sums everything (None), nothing
   (empty list: copy) or along dim 0 of a 2-D tensor */
void atg_sum_dim_intlist(tensor *out__, tensor self, int64_t *dim_data, int dim_len, int keepdim, int dtype) {
  mock_tensor *s = self;
  (void)keepdim;
  last_dim_null = dim_data == NULL;
  last_dim_len = dim_len;
  int kind = dtype < 0 ? s->kind : dtype;
  if (dim_data == NULL) {
    mock_tensor *m = mk(0, NULL, kind);
    for (int64_t i = 0; i < s->numel; i++) m->data[0] += s->data[i];
    out__[0] = m;
  } else if (dim_len == 0) {
    mock_tensor *m = copy_of(s); m->kind = kind; out__[0] = m;
  } else {
    if (s->ndim != 2 || dim_data[0] != 0) { set_err("mock: sum_dim_intlist supports dim 0 of 2-D"); return; }
    int64_t dims[1] = { s->dims[1] };
    mock_tensor *m = mk(1, dims, kind);
    for (int64_t r = 0; r < s->dims[0]; r++)
      for (int64_t c = 0; c < s->dims[1]; c++) m->data[c] += s->data[r * s->dims[1] + c];
    out__[0] = m;
  }
}

/* records the string and the optional double; returns a copy */
void atg_pad(tensor *out__, tensor self, int64_t *pad_data, int pad_len, char *mode_ptr, int mode_len, double value_v, uint8_t value_null) {
  (void)pad_data; (void)pad_len;
  int n = mode_len < 63 ? mode_len : 63;
  memcpy(last_mode, mode_ptr, n);
  last_mode[n] = 0;
  last_value_null = value_null;
  last_value = value_v;
  out__[0] = copy_of(self);
}

void atg_to(tensor *out__, tensor self, int device) { (void)device; out__[0] = copy_of(self); }
void atg_totype(tensor *out__, tensor self, int scalar_type) {
  mock_tensor *m = copy_of(self); m->kind = scalar_type; out__[0] = m;
}
