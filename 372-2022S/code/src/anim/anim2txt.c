
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include "anim_dev.h"

/* The maximum number of scalar values that will be printed in one
   line */
const int MAX_LEN = 10;

/* A range represents a regular set of nonnegative integers, i.e., a
   set of the form lo, lo+stride, lo+2*stride, ....  The set can be
   either bounded or unbounded. */
struct range {
  size_t lo;
  size_t hi;  // UNDEF indicates unbounded
  size_t stride;
};

/* Different options for formatting scalar values: scientific
   notation, fixed point floats, or raw data. */
enum format { EXP, FIX, RAW };

struct param {
  ANIM_File af;
  FILE * out;
  struct range frames;
  struct range domains[3];
  bool debug;
  bool check;
  bool pure;
  bool show_info;
  enum format format;
};

#define UNDEF ((size_t)(-1))
const struct range null_range = {UNDEF, UNDEF, UNDEF};

// TODO: implement -c
// TODO: implement -d
// TODO: -h should not return non-0 error code
static void quit() {
  printf("\
Usage: anim2txt [OPTIONS] filename                                     \n\
Displays the information in an ANIM file as human-readable text.       \n\
Options:                                                               \n\
  -c        : check integrity of the entire file                       \n\
  -d        : debug: print lots of messages                            \n\
  -e        : show floating point numbers using scientific notation    \n\
  -f<RANGE> : print only frames in the range <RANGE>                   \n\
  -h        : print this help message and quit                         \n\
  -i        : print file info only                                     \n\
  -n        : do not print file info                                   \n\
  -o OUTFILE: send output to file named OUTFILE (default is stdout)    \n\
  -p        : pure data: don't print \"Frame...\" lines                \n\
  -r        : show the raw file data, not floating point numbers       \n\
  -x<RANGE> : print only points with x-coordinate in <RANGE>	       \n\
  -y<RANGE> : print only points with y-coordinate in <RANGE>           \n\
  -z<RANGE> : print only points with z-coordinate in <RANGE>           \n\
                                                                       \n\
<RANGE> has one of the following forms:                                \n\
  N         : the set consisting of a single natural number N          \n\
  M-N       : the set M, M+1, ..., N                                   \n\
  M-        : the set M, M+1, ...                                      \n\
  M-#P      : the set M, M+P, M+2*P, ....  (P must be positive)        \n\
  M-N#P     : the set M, M+P, M+2*P, ..., intersected with {n|n<=N}    \n\
");
  exit(1);
}

static bool range_is_null(struct range r) {
  return r.stride == UNDEF;
}

static inline bool range_is_unbounded(struct range r) {
  return !range_is_null(r) && r.hi == UNDEF;
}

static inline bool range_is_empty(struct range r) {
  return !range_is_null(r) && r.hi != UNDEF && r.lo > r.hi;
}

static struct range make_range(size_t lo, size_t hi, size_t stride) {
  assert(lo != UNDEF && hi != UNDEF && stride != 0 && stride != UNDEF);
  return (struct range){lo, hi, stride};
}
 
static struct range make_unbounded_range(size_t lo, size_t stride) {
  assert(lo != UNDEF && stride != 0 && stride != UNDEF);
  return (struct range){lo, UNDEF, stride};
}

static struct range make_range_long(long lo, long hi, long stride) {
  assert(lo >= 0 && hi >= 0 && stride > 0);
  return make_range((size_t)lo, (size_t)hi, (size_t)stride);
}

static struct range make_unbounded_range_long(long lo, long stride) {
  assert(lo >= 0 && stride > 0);
  return make_unbounded_range((size_t)lo, (size_t)stride);
}

static inline size_t range_size(struct range r) {
  assert(!range_is_null(r));
  if (range_is_empty(r)) return 0;
  // lo, lo+p, lo+2p, ..., lo+kp.  hi-lo-p < kp <= hi-lo.  k=(hi-lo)/p.
  return 1 + (r.hi-r.lo)/r.stride;
}

static struct range parse_range(const char * s) {
  long lo, hi, stride; // parsing produces longs
  int len = strlen(s), dash = len, hash = len, idx;

  // first, find indexes of - and # (if they exist)...
  for (idx = 0; idx < len; idx++) {
    const char c = s[idx];
    if (c == '-') {
      dash = idx++;
      break;
    }
    if (c < '0' || c > '9') quit();
  }
  for (; idx < len; idx++) {
    const char c = s[idx];
    if (c == '#') {
      hash = idx++;
      break;
    }
    if (c < '0' || c > '9') quit();
  }
  for (; idx < len; idx++) {
    const char c = s[idx];
    if (c < '0' || c > '9') quit();
  }
  if (dash < 1) quit();  // s can't start with '-' and len must be at least 1
  // next, compute lo, hi, and stride...
  lo = atol(s);
  if (lo < 0) quit();
  if (dash == len) { // no dash: s must be a single integer N
    assert(hash == len);
    hi = lo;
    stride = 1;
  } else { // s has form M-...
    if (dash < hash - 1) { // M-N...: hi is specified
      hi = atol(s + dash + 1);
      if (hi < 0 || lo > hi) quit();
    } else hi = -1; // hi is unspecified: unbounded
    // now process the #...
    if (hash < len - 1) { // s ends with #M
      stride = atol(s + hash + 1);
      if (stride <= 0) quit();
    } else if (hash == len - 1) quit(); // s can't end with #
    else stride = 1; // hash == len: there is no #
  }
  return hi >= 0 ? make_range_long(lo, hi, stride) :
    make_unbounded_range_long(lo, stride);
}

/* Compute the 1d-index of a multi-dimensional array element, given
   its multi-dimensional index, assuming row-major order.  dim is the
   dimension, pos is an array of length dim giving the multi-d indexes
   into the array, and len is an array of length dim giving the array
   length in each dimension.  Note ethat len[0] is not needed or used
   (unless you want to do bounds checking, which we don't).  This
   function assumes 0<=pos[i]<len[i] for 0<=i<dim.  Behavior is
   undefined if this is not the case. */
static inline size_t get_offset(int dim, size_t * pos, int * len) {
  size_t idx = 0, l = 1;
  int i = dim-1;

  while (1) {
    idx += pos[i]*l;
    l *= len[i];
    i--;
    if (i<0) break;
  }
#ifdef DEBUG
  printf("get_offset: dim=%d, pos=(%zu, %zu), len=(%d, %d), result: %zu\n",
	 dim, pos[0], pos[1], len[0], len[1], idx);
#endif  
  return idx;
}

static void init_pos(int dim, struct range * domains, size_t * pos) {
  for (int i=0; i<dim-1; i++) pos[i] = domains[i].lo;
  pos[dim-1] = 0;
}

/* Increments pos to point to the next row in the currrent frame,
   within the set specified by domains.  If there is no such row,
   false is returned, otherwise true is returned.  Note that
   pos[dim-1] should always be 0, since pos points to the beginning of
   a row.  */
static bool next_row_in_frame(int dim, struct range * domains, size_t * pos) {
  for (int i=dim-2; i>=0; i--) {
    pos[i] += domains[i].stride;
    if (pos[i] <= domains[i].hi) return true;
    else pos[i] = domains[i].lo;
  }
  return false;
}

static void init_frame_pos(struct param * pp, size_t * frame, size_t * pos) {
  const ANIM_File af = pp->af;
  
  *frame = pp->frames.lo;
  init_pos(af->dim, pp->domains, pos);
  ANIM_Seek_frame(af, *frame);

  size_t nread = ANIM_Read_next(af);

  if (nread != af->frame_size) {
    fprintf(stderr,
	    "anim: error reading frame %zu: %zu values read, expected %zu\n",
	    *frame, nread, af->frame_size);
    fflush(stderr);
    exit(1);
  }
}

/* Proceeds to next row specified by pp, if there is one.  The next
   row may involving reading in the next frame, which this function
   does.  Returns true, if there is a next row, in which case pos and
   frame have been updated correctly (and possibly a new frame read
   in).  Otherwise, returns false. */
static bool next_row(struct param * pp, size_t * frame, size_t * pos) {
  const ANIM_File af = pp->af;
  const int dim = af->dim;
  bool ok = next_row_in_frame(dim, pp->domains, pos);

  if (ok) return true;
  init_pos(dim, pp->domains, pos);
  *frame += pp->frames.stride;
  if (*frame > pp->frames.hi) return false;
  ANIM_Seek_frame(af, *frame);

  size_t nread = ANIM_Read_next(af);

  if (nread != af->frame_size) {
    fprintf(stderr,
	    "anim: error reading frame %zu: %zu values read, expected %zu\n",
	    *frame, nread, af->frame_size);
    fflush(stderr);
    exit(1);
  }
  return true;
}

/* Consumes a param pointer pp and produces the output in the case
   where kind is ANIM_HEAT or ANIM_GRAPH.  pp->frames will be a
   non-null, bounded range. */
static void heat_or_graph(struct param * pp) {
  const ANIM_File af = pp->af;
  const int dim = ANIM_Get_dim(af);
  int * len = af->lengths;
  FILE * const out = pp->out;

#ifdef DEBUG
  printf("len = (%d, %d, %d)\n", len[0], len[1], len[2]);
#endif

  if (range_is_empty(pp->frames)) return; // nothing to do
  if (dim == 1 && !range_is_null(pp->domains[1]))
    fprintf(stderr, "anim: warning: ignoring y range for 1d problem\n");
  if (dim <= 2 && !range_is_null(pp->domains[2]))
    fprintf(stderr, "anim: warning: ignoring z range for %dd problem\n", dim);
  for (int i=0; i<dim; i++) {
    struct range r = pp->domains[i];

    if (range_is_null(r))
      pp->domains[i] = make_range(0, len[i] - 1, 1);
    else if (range_is_unbounded(r))
      pp->domains[i].hi = len[i] - 1;
    else if (r.hi >= len[i]) {
      fprintf(stderr,
	      "anim: warning: %s upper bound %zu exceeds maximum %d\n",
	      (i==0 ? "x" : i==1 ? "y" : "z"), r.hi, len[i]-1);
      fflush(stderr);
      pp->domains[i].hi = len[i] - 1;
    }
    if (range_is_empty(pp->domains[i]))
      return; // nothing to do

#ifdef DEBUG
    r = pp->domains[i];
    printf("lo=%zu hi=%zu stride=%zu\n", r.lo, r.hi, r.stride);
    fflush(stdout);
#endif
  }

  const size_t row_size = range_size(pp->domains[dim-1]);
  const size_t row_lo = pp->domains[dim-1].lo, row_hi = pp->domains[dim-1].hi,
    row_stride = pp->domains[dim-1].stride;
  const double min = af->ranges[dim].min, max = af->ranges[dim].max;
  size_t frame, pos[dim];
  
  init_frame_pos(pp, &frame, pos);
  do {
    // print row header:
    if (!pp->pure) {
      fprintf(out, "Frame %4ld", frame);
      if (dim >= 2) fprintf(out, ", x=%ld", pos[0]);
      if (dim >= 3) fprintf(out, ", y=%ld", pos[1]);
      fprintf(out, ":");
      if (row_size <= MAX_LEN) {
	fprintf(out, " ");
      } else {
	fprintf(out, "\n");
      }
    }

    const size_t offset = get_offset(dim, pos, len);

#ifdef DEBUG
    printf("pos = (%zu %zu %zu)    offset = %zu\n",
	   pos[0], pos[1], pos[2], offset);
    fflush(stdout);
#endif

#define PRINT_ROW(T, x)						\
    const T * const row = (const T *)(af->buf) + offset;	\
    unsigned c = 0;						\
    if (pp->format == RAW)					\
      for (size_t i=row_lo; i<=row_hi; i+=row_stride) {		\
      	if (c == MAX_LEN) { c = 0; fprintf(out, "\n"); }	\
	fprintf(out, "%4u", row[i]);				\
	c++;							\
      }								\
    else if (pp->format == FIX)					\
      for (size_t i=row_lo; i<=row_hi; i+=row_stride) {		\
	if (c == MAX_LEN) { c = 0; fprintf(out, "\n"); }	\
	fprintf(out, "%8.2lf ", x);				\
	c++;							\
      }								\
    else							\
      for (size_t i=row_lo; i<=row_hi; i+=row_stride) {		\
	if (c == MAX_LEN) { c = 0; fprintf(out, "\n"); }	\
	fprintf(out, "%10.2le ", x);				\
	c++;							\
      }								\
    fprintf(out, "\n")
    
    if (af->kind == ANIM_HEAT) {
      PRINT_ROW(ANIM_byte, ANIM_byte_to_double(row[i], min, max));
    } else {
      const int n = len[dim];
      PRINT_ROW(ANIM_coord_t, ANIM_coord_to_double(row[i], n, min, max));
    }
  } while (next_row(pp, &frame, pos)); 
}

void nbody(struct param * pp) {
  const ANIM_File af = pp->af;
  const int dim = ANIM_Get_dim(af);
  FILE * const out = pp->out;
  struct range const frames = pp->frames;
  ANIM_coord_t * const buf = (ANIM_coord_t*) af->buf;
  size_t rv, frame_size = af->frame_size;
  int nbodies = af->nbodies;
  bool pure = pp->pure;
  const int * const len = (const int *)af->lengths;
  const ANIM_range_t * const ranges = (const ANIM_range_t *)af->ranges;
  
  if (range_is_empty(frames)) return; // nothing to do
  for (int i=0; i<3; i++) {
    if (!range_is_null(pp->domains[i])) {
      fprintf(stderr,
	      "anim: warning: %s specification not used in nbody animation\n",
	      (i==0 ? "x" : i==1 ? "y" : "z"));
      fflush(stderr);
    }
  }
  for (size_t frame = frames.lo; frame <= frames.hi; frame += frames.stride) {
    ANIM_Seek_frame(af, frame);
    rv = ANIM_Read_next(af);
    if (rv != frame_size) {
      fprintf(stderr,
	      "anim: error reading frame %zu: read %zu values, expected %zu\n",
	      frame, rv, frame_size);
      fflush(stderr);
      exit(1);
    }
    if (!pure) fprintf(out, "Frame %zu:\n", frame);
    for (int i=0, count=0; i<nbodies; i++) {
      if (!pure) fprintf(out, "%6d. ", i);
      for (int j=0; j<dim; j++)
	if (pp->format == FIX)
	  fprintf(out, "%16.2lf ",
		  ANIM_coord_to_double(buf[count++], len[j], ranges[j].min, ranges[j].max));
	else if (pp->format == EXP)
	  fprintf(out, "%14.2le ",
		  ANIM_coord_to_double(buf[count++], len[j], ranges[j].min, ranges[j].max));
	else
	  fprintf(out, "%8u ", buf[count++]);
      fprintf(out, "\n");
    }
  }
}

int main(int argc, char * argv[]) {
  struct param param;
  char * out_fn = NULL, * anim_fn = NULL;

  //  ANIM_Set_debug(); // useful when debugging
  param.frames = null_range;
  param.domains[0] = param.domains[1] = param.domains[2] = null_range;
  param.pure = false;
  param.debug = false;
  param.check = false;
  param.show_info = true;
  param.format = FIX;
  for (int i=1; i<argc; i++) {
    char * arg = argv[i];
    const char first = arg[0];

    if (first != '-') {
      if (anim_fn != NULL) quit();
      anim_fn = arg;
    } else if (strcmp(arg, "-c") == 0) {
      param.check = true;
    } else if (strcmp(arg, "-d") == 0) {
      param.debug = true;
    } else if (strcmp(arg, "-e") == 0) {
      param.format = EXP;
    } else if (arg[1] == 'f') {
      param.frames = parse_range(arg+2);
    } else if (strcmp(arg, "-h") == 0) {
      quit();
    } else if (strcmp(arg, "-i") == 0) {
      param.frames = make_range(1, 0, 1);
    } else if (strcmp(arg, "-n") == 0) {
      param.show_info = false;
    } else if (strcmp(arg, "-o") == 0) {
      if (++i >= argc) quit();
      out_fn = argv[i];
    } else if (strcmp(arg, "-p") == 0) {
      param.pure = true;
      param.show_info = false;
    } else if (strcmp(arg, "-r") == 0) {
      param.format = RAW;
    } else if (arg[1] == 'x') {
      param.domains[0] = parse_range(arg+2);
    } else if (arg[1] == 'y') {
      param.domains[1] = parse_range(arg+2);
    } else if (arg[1] == 'z') {
      param.domains[2] = parse_range(arg+2);
    } else quit();    
  }
  if (param.debug) ANIM_Set_debug();
  if (anim_fn == NULL) quit();
  if (out_fn == NULL) {
    param.out = stdout;
  } else {
    param.out = fopen(out_fn, "w");
    if (param.out == NULL) {
      fprintf(stderr, "anim: unable to open file %s for writing.\n",
	      out_fn);
      fflush(stderr);
      exit(1);
    }
  }
  param.af = ANIM_Open(anim_fn);
  assert(param.af);
  if (param.show_info) {
    ANIM_Print_metadata(param.out, param.af);
  }
  if (range_is_null(param.frames))
    param.frames = (struct range){0, param.af->nframes - 1, 1};
  else if (range_is_unbounded(param.frames))
    param.frames.hi = param.af->nframes - 1;
  else if (param.frames.hi >= param.af->nframes) {
    fprintf(stderr, "anim: warning: frame id %zu exceeds maximum %zu\n",
	    param.frames.hi, param.af->nframes - 1);
    fflush(stderr);
    param.frames.hi = param.af->nframes - 1;
  }
  if (param.af->kind == ANIM_HEAT || param.af->kind == ANIM_GRAPH) {
    heat_or_graph(&param);
  } else { // NBODY
    nbody(&param);
  }
  ANIM_Close(param.af);
  fclose(param.out);
}
