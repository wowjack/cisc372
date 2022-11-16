/* Filename : anim.c
   Author   : Stephen F. Siegel
   Created  : 23-jun-2020
   Modified : 23-jun-2020

   Implementation of ANIM library.  See anim.h.
   
   We use C's stdio library for file manipulation.  An ANIM file is
   a binary file with a header containing file metadata followed by
   the sequence of frames, each of which is a sequence of some scalar
   type, currently either unsigned chars (for HEAT), or unsigned shorts
   (for GRAPH and NBODY).

   The file header has the following format:

   kind code (int)
   dim (int)
   lengths (int[3])
   ranges (int[4])
   nframes (size_t)
   frame_size (size_t)

   FOR NBODY files only, this is followed by:
     nbodies (int)
     radii (int[nbodies])
     ncolors (int)
     colors (ANIM_color_t[ncolors])
     bcolors (int[nbodies])

   Each data frame is a sequence of frame_size scalar values.  For
   HEAT: each scalar value is a byte, representing a shade of red/blue
   For GRAPH: each scalar value is an unsigned short, in
   0..lengths[dim]-1, a coordinate.  For NBODIES: each scalar value is
   an unsigned short; they come in tuples representing coordinates,
   measured in pixels.

   Note that in ANIM, raw file data that are coordinates in NBODY have
   y values that increase as you go up (the origin is in the lower
   left hand corner).  When an ANIM file is converted to GIF, the
   y-coordinates must be flipped.
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#include "anim_dev.h"

#define ANIM_STATUS_BAR "0%....................25%......................50%\
......................75%.....................100%"

/* Are we printing debugging messages? */
_Bool debugging = false;

char * ANIM_kind_to_str(ANIM_Kind kind) {
  switch(kind) {
  case ANIM_HEAT:
    return "ANIM_HEAT";
  case ANIM_GRAPH:
    return "ANIM_GRAPH";
  case ANIM_NBODY:
    return "ANIM_NBODY";
  default:
    assert(false);
  }
}

int ANIM_kind_to_int(ANIM_Kind kind) {
  switch(kind) {
  case ANIM_HEAT:
    return 1;
  case ANIM_GRAPH:
    return 2;
  case ANIM_NBODY:
    return 3;
  default:
    assert(false);
  }
}

int ANIM_int_to_kind(int val) {
  switch (val) {
  case 1:
    return ANIM_HEAT;
  case 2:
    return ANIM_GRAPH;
  case 3:
    return ANIM_NBODY;
  default:
    assert(false);
  }
}

void ANIM_Set_debug() {
  debugging = true;
}

void ANIM_Unset_debug() {
  debugging = false;
}

ANIM_color_t ANIM_Make_color(ANIM_byte red, ANIM_byte green, ANIM_byte blue) {
  return ((ANIM_color_t)red)<<16 | ((ANIM_color_t)green)<<8
    | ((ANIM_color_t)blue);
}

ANIM_byte ANIM_Get_red(ANIM_color_t color) {
  return (ANIM_byte)(color>>16);
}

ANIM_byte ANIM_Get_green(ANIM_color_t color) {
  return (ANIM_byte)((color>>8) & 255u);
}

ANIM_byte ANIM_Get_blue(ANIM_color_t color) {
  return (ANIM_byte)(color & 255u);
}

void ANIM_Print_int_array(FILE * out, int n, int * a) {
  fprintf(out, "{");
  for (int i=0; i<n; i++) {
    if (i>0) fprintf(out, ", ");
    fprintf(out, "%d", a[i]);
  }
  fprintf(out, "}");
}

void ANIM_Print_range(FILE * out, ANIM_range_t range) {
  fprintf(out, "[%lf,%lf]", range.min, range.max);
}

void ANIM_Print_range_array(FILE * out, int n, ANIM_range_t * a) {
  fprintf(out, "{");
  for (int i=0; i<n; i++) {
    if (i>0) fprintf(out, ", ");
    ANIM_Print_range(out, a[i]);
  }
  fprintf(out, "}");
}

void ANIM_Print_metadata(FILE * out, ANIM_File af) {
  int dim = af->dim;
  
  fprintf(out, "%s (kind = %s, dim = %d):",
	  af->filename, ANIM_kind_to_str(af->kind), dim);
  //printf("\n  open for %sing", (af->read ? "read" : "writ"));
  fprintf(out, "\n  lengths = ");
  ANIM_Print_int_array(out, (af->kind == ANIM_GRAPH ? dim+1 : dim), af->lengths );
  fprintf(out, "\n  ranges = ");
  ANIM_Print_range_array(out, (af->kind == ANIM_NBODY ? dim : dim+1), af->ranges );
  fprintf(out, "\n  nframes = %zu", af->nframes);
  fprintf(out, "\n  frame_size = %zu", af->frame_size);
  //printf("\n  frame_count = %zu", af->framecount);
  //printf("\n  buf = %p", af->buf);
  if (af->kind == ANIM_NBODY) {
    fprintf(out, "\n  nbodies = %d", af->nbodies);
    fprintf(out, "\n  ncolors = %d", af->ncolors);
    fprintf(out, "\n  colors:");
    for (int i=0; i<af->ncolors; i++) {
      ANIM_color_t color = af->colors[i];
      
      fprintf(out, "\n    color[%d] = (%u, %u, %u)", i, ANIM_Get_red(color),
	      ANIM_Get_green(color), ANIM_Get_blue(color));
    }
    fprintf(out, "\n  bodies:");
    for (int i=0; i<af->nbodies; i++) {
      fprintf(out, "\n    body %d: radius = %d, color = %d", i,
	      af->radii[i], af->bcolors[i]);
    }
  }
  printf("\n");
}

/* Computes number of scalars in one frame of a HEAT or GRAPH animation */
static inline size_t frame_size_heat_graph(int dim, int * lengths) {
  size_t frame_size = 1;

  for (int i=0; i<dim; i++) frame_size *= lengths[i];
  return frame_size;
}

/* Computes number of scalars in one frame of an NBODY animation */
static inline size_t frame_size_nbody(int dim, int nbodies) {
  return nbodies*dim;
}

/* Computes the size of the buffer, in bytes, required to hold one
   frame in an animation. */
static inline size_t buffer_size(ANIM_Kind kind, size_t frame_size) {
  return kind == ANIM_HEAT ? frame_size : frame_size * sizeof(ANIM_coord_t);
}

static ANIM_File
create_common(ANIM_Kind kind, int dim, int nlengths, int * lengths,
	      int nranges, ANIM_range_t * ranges, size_t frame_size,
	      char * filename) {
  ANIM_File af = (ANIM_File)malloc(sizeof(struct ANIM_File_s));
  
  assert(af);
  af->kind = kind;
  af->read = false;
  af->dim = dim;
  for (int i=0; i<nlengths; i++) {
    assert(lengths[i] >= 1);
    af->lengths[i] = lengths[i];
  }
  for (int i=nlengths; i<3; i++)
    af->lengths[i] = 0;
  for (int i=0; i<nranges; i++) {
    assert(ranges[i].min < ranges[i].max);
    af->ranges[i] = ranges[i];
  }
  for (int i=nranges; i<4; i++)
    af->ranges[i] = (ANIM_range_t){0.0,0.0};
  af->nframes = 0; // set this when file is closing
  af->frame_size = frame_size;
  af->framecount = 0;
  assert(filename);
  af->filename = malloc(strlen(filename)+1);
  assert(af->filename);
  strcpy(af->filename, filename);
  af->file = fopen(filename, "wb");
  assert(af->file);
  af->buf = malloc(buffer_size(kind, frame_size));
  assert(af->buf);
  // kind NBODY can overwrite this part
  af->nbodies = 0;
  af->radii = NULL;
  af->ncolors = 0;
  af->colors = NULL;
  af->bcolors = NULL;
  // end NBODY part

  int kind_code = ANIM_kind_to_int(af->kind);
  size_t rv = fwrite(&kind_code, sizeof(int), 1, af->file);
  
  assert(rv == 1);
  rv = fwrite(&dim, sizeof(int), 1, af->file);
  assert(rv == 1);
  rv = fwrite(af->lengths, sizeof(int), 3, af->file);
  assert(rv == 3);
  rv = fwrite(af->ranges, sizeof(ANIM_range_t), 4, af->file);
  assert(rv == 4);
  rv = fwrite(&af->nframes, sizeof(size_t), 1, af->file);
  assert(rv == 1);
  rv = fwrite(&af->frame_size, sizeof(size_t), 1, af->file);
  assert(rv == 1);
  if (debugging) {
    printf("File opened and header written.\n");
    fflush(stdout);
  }
  return af;
}

/* Computes the number of bytes written in the header part of the file
   by function create_common().  Keep this consistent with that
   function. */
static inline size_t basic_header_size() {
 return 5*sizeof(int) + 4*sizeof(ANIM_range_t) + 2*sizeof(size_t);
}

/* Returns the offset, measured in bytes, in the file, of the nframes
   field.  Keep this consistent with create_common(). */
long int ANIM_Get_nframes_offset() {
  return 5*sizeof(int) + 4*sizeof(ANIM_range_t);
}
		       
ANIM_File ANIM_Create_heat(int dim, int lengths[dim],
			   ANIM_range_t ranges[dim+1],
			   char * filename) {
  ANIM_File af = create_common(ANIM_HEAT, dim, dim, lengths, dim+1, ranges,
			       frame_size_heat_graph(dim, lengths), filename);

  if (debugging) {
    printf("Opening for writing: ");
    ANIM_Print_metadata(stdout, af);
    fflush(stdout);
  }
  return af;
}

ANIM_File ANIM_Create_graph(int dim, int lengths[dim+1],
			    ANIM_range_t ranges[dim+1],
			    char * filename) {
  ANIM_File af = create_common(ANIM_GRAPH, dim, dim+1, lengths, dim+1, ranges,
			       frame_size_heat_graph(dim, lengths), filename);
  
  if (debugging) {
    printf("Opening for writing: ");
    ANIM_Print_metadata(stdout, af);
    fflush(stdout);
  }
  return af;
}

ANIM_File ANIM_Create_nbody(int dim, int lengths[dim],
			    ANIM_range_t ranges[dim],
			    int nbodies, int radii[nbodies],
			    int ncolors,  ANIM_color_t colors[ncolors],
			    int bcolors[nbodies], char * filename) {
  ANIM_File af =
    create_common(ANIM_NBODY, dim, dim, lengths, dim, ranges,
		  frame_size_nbody(dim, nbodies), filename);
  size_t rv;

  assert(nbodies >= 1);
  af->nbodies = nbodies;
  af->radii = malloc(nbodies * sizeof(int));
  assert(af->radii);
  for (int i=0; i<nbodies; i++) af->radii[i] = radii[i];
  assert(ncolors >= 1);
  af->ncolors = ncolors;
  af->colors = malloc(ncolors*sizeof(ANIM_color_t));
  assert(af->colors);
  for (int i=0; i<ncolors; i++) af->colors[i] = colors[i];
  af->bcolors = malloc(nbodies*sizeof(int));
  assert(af->bcolors);
  for (int i=0; i<nbodies; i++) af->bcolors[i] = bcolors[i];
  // write the metadata to file...
  rv = fwrite(&nbodies, sizeof(int), 1, af->file);
  assert(rv == 1);
  rv = fwrite(radii, sizeof(int), nbodies, af->file);
  assert(rv == nbodies);
  rv = fwrite(&ncolors, sizeof(int), 1, af->file);
  assert(rv == 1);
  rv = fwrite(colors, sizeof(ANIM_color_t), ncolors, af->file);
  assert(rv == ncolors);
  rv = fwrite(bcolors, sizeof(int), nbodies, af->file);
  assert(rv == nbodies);

  if (debugging) {
    printf("Opening for writing: ");
    ANIM_Print_metadata(stdout, af);
    fflush(stdout);
  }

  return af;
}

size_t ANIM_Heat_file_size(int dim, int lengths[dim], size_t nframes) {
  return basic_header_size() +
    nframes*buffer_size(ANIM_HEAT, frame_size_heat_graph(dim, lengths));
}

size_t ANIM_Graph_file_size(int dim, int lengths[dim], size_t nframes) {
  return basic_header_size() +
    nframes*buffer_size(ANIM_GRAPH, frame_size_heat_graph(dim, lengths));
}

size_t ANIM_Nbody_file_size(int dim, int nbodies, int ncolors, size_t nframes) {
  // in addition to the basic header:
  // nbodies (int), radii (nbodies int's), ncolors (int),
  // colors (ncolors ANIM_color_t), bcolors (nbodies int's).
  size_t header = basic_header_size() + sizeof(int)
    + nbodies*sizeof(int) + sizeof(int)
    + ncolors*sizeof(ANIM_color_t) + nbodies*sizeof(int);

  return header +
    nframes*buffer_size(ANIM_NBODY, frame_size_nbody(dim, nbodies));
}

size_t ANIM_File_size(ANIM_File af, size_t nframes) {
  switch (af->kind) {
  case ANIM_HEAT:
    return ANIM_Heat_file_size(af->dim, af->lengths, nframes);
  case ANIM_GRAPH:
    return ANIM_Graph_file_size(af->dim, af->lengths, nframes);
  case ANIM_NBODY:
    return ANIM_Nbody_file_size(af->dim, af->nbodies, af->ncolors, nframes);
  default:
    assert(false);
  }
}

static size_t write_frame_heat(ANIM_File af, double * buf) {
  int dim = af->dim;
  size_t size = af->frame_size;
  unsigned char * color_buf = af->buf;
  const double min = af->ranges[dim].min, max = af->ranges[dim].max;

  if (debugging) {
    printf("Writing frame %zu to %s:\n", af->framecount, af->filename);
    fflush(stdout);
    for (size_t i=0; i<size; i++) {
      const double val = buf[i];

      printf("%7.2lf ", val);
      color_buf[i] = ANIM_double_to_byte(val, min, max);
    }
    printf("\n");
    fflush(stdout);
  } else {
    for (size_t i=0; i<size; i++)
      color_buf[i] = ANIM_double_to_byte(buf[i], min, max);
  }
  
  size_t rv = fwrite(color_buf, 1, size, af->file);

  assert(rv == size);
  af->framecount++;
  return rv;
}

// TODO: debugging should print more stuff here...

static size_t write_frame_graph(ANIM_File af, double * buf) {
  const int dim = af->dim;
  const size_t size = af->frame_size;
  ANIM_coord_t * const coord_buf = af->buf;
  const double min = af->ranges[dim].min, max = af->ranges[dim].max;
  const int n = af->lengths[dim];

  for (size_t i=0; i<size; i++)
    coord_buf[i] = ANIM_double_to_coord(buf[i], n, min, max);

  size_t rv = fwrite(coord_buf, sizeof(ANIM_coord_t), size, af->file);

  assert(rv == size);
  af->framecount++;
  return rv;
}

static size_t write_frame_nbody(ANIM_File af, double * buf) {
  int dim = af->dim;
  size_t size = af->frame_size;
  ANIM_coord_t * const coord_buf = af->buf;

  for (int i=0; i<dim; i++) {
    const double min = af->ranges[i].min, max = af->ranges[i].max;
    const int n = af->lengths[i];
      
    for (size_t j=i; j<size; j += dim)
      coord_buf[j] = ANIM_double_to_coord(buf[j], n, min, max);
  }

  if (debugging) {
    printf("\nANIM: writing buffer:\n");
    if (dim == 1) {
      for (size_t j=0; j<size; j+=1)
	printf("%zu. (%lf) --> (%u)\n", j, buf[j], coord_buf[j]);
    } else if (dim == 2) {
      for (size_t j=0; j<size; j+=2)
	printf("%zu. (%lf, %lf) --> (%u, %u)\n",
	       j, buf[j], buf[j+1], coord_buf[j], coord_buf[j+1]);
    } else {
      for (size_t j=0; j<size; j+=3)
	printf("%zu. (%lf, %lf, %lf) --> (%u, %u, %u)\n",
	       j, buf[j], buf[j+1], buf[j+2],
	       coord_buf[j], coord_buf[j+1], coord_buf[j+2]);
    }
    fflush(stdout);
  }
  
  size_t rv = fwrite(coord_buf, sizeof(ANIM_coord_t), size, af->file);

  assert(rv == size);
  af->framecount++;
  return rv;
}

size_t ANIM_Write_frame(ANIM_File af, double * buf) {
  switch(af->kind) {
  case ANIM_HEAT:
    return write_frame_heat(af, buf);
  case ANIM_GRAPH:
    return write_frame_graph(af, buf);
  case ANIM_NBODY:
    return write_frame_nbody(af, buf);
  default:
    assert(false);
  }
}

/* Opens a file for reading and reads the header.  If verbose is true,
   messages are printed to stdout to help with debugging. */
static ANIM_File open(char * filename, bool verbose) {
  assert(filename);
  
  int filename_length = strlen(filename);

  if (filename_length <= 0 || filename_length > 1000) {
    fprintf(stderr, "anim: illegal filename, length = %d\n", filename_length);
    fflush(stderr);
    exit(1);
  }
  if (verbose) printf("anim: opening %s\n", filename);
  
  ANIM_File af = malloc(sizeof(struct ANIM_File_s));
  size_t rv;

  assert(af);
  af->file = fopen(filename, "rb");
  if (af->file == NULL) {
    fprintf(stderr, "anim: could not open file %s\n", filename);
    fflush(stderr);
    exit(1);
  }
  af->read = true;

  int kind_code;
  
  rv = fread(&kind_code, sizeof(int), 1, af->file);
  if (rv != 1) {
    fprintf(stderr, "anim: malformed file %s: no kind code in header\n",
	    filename);
    fflush(stderr);
    exit(1);
  }
  af->kind = ANIM_int_to_kind(kind_code);
  if (verbose) printf("anim: kind = %s\n", ANIM_kind_to_str(af->kind));
  rv = fread(&af->dim, sizeof(int), 1, af->file);
  if (rv != 1) {
    fprintf(stderr, "anim: malformed file %s: no dim in header\n",
	    filename);
    fflush(stderr);
    exit(1);
  }
  if (verbose) printf("anim: dim = %d\n", af->dim);
  if (af->dim < 1 || af->dim > 3) {
    fprintf(stderr, "anim: illegal dimension in %s: %d\n",
	    filename, af->dim);
    fflush(stderr);
    exit(1);
  }
  rv = fread(af->lengths, sizeof(int), 3, af->file);
  if (verbose) {
    printf("anim: read lengths = ");
    ANIM_Print_int_array(stdout, rv, af->lengths);
    printf("\n");
  }
  assert(rv == 3);
  rv = fread(af->ranges, sizeof(ANIM_range_t), 4, af->file);
  if (verbose) {
    printf("anim: read ranges = ");
    ANIM_Print_range_array(stdout, rv, af->ranges);
    printf("\n");
  }
  assert(rv == 4);
  rv = fread(&af->nframes, sizeof(size_t), 1, af->file);
  if (verbose) printf("anim: read nframes = %zu\n", af->nframes);
  assert(rv == 1);
  rv = fread(&af->frame_size, sizeof(size_t), 1, af->file);
  if (verbose) printf("anim: read frame_size = %zu\n", af->frame_size);
  assert(rv == 1);
  if (af->kind == ANIM_NBODY) {
    rv = fread(&af->nbodies, sizeof(int), 1, af->file);
    assert(rv == 1);
    assert(af->nbodies >= 1);
    af->radii = malloc(af->nbodies*sizeof(int));
    assert(af->radii);
    rv = fread(af->radii, sizeof(int), af->nbodies, af->file);
    assert(rv == af->nbodies);
    rv = fread(&af->ncolors, sizeof(int), 1, af->file);
    assert(rv == 1);
    assert(af->ncolors >= 1);
    af->colors = malloc(af->ncolors*sizeof(ANIM_color_t));
    assert(af->colors);
    rv = fread(af->colors, sizeof(ANIM_color_t), af->ncolors, af->file);
    assert(rv == af->ncolors);
    af->bcolors = malloc(af->nbodies*sizeof(int));
    assert(af->bcolors);
    rv = fread(af->bcolors, sizeof(int), af->nbodies, af->file);
    assert(rv == af->nbodies);
  }  
  af->framecount = 0;
  af->filename = malloc(filename_length + 1);
  assert(af->filename);
  strcpy(af->filename, filename);

  const size_t bufsz =
    af->kind == ANIM_HEAT ? af->frame_size : af->frame_size * sizeof(ANIM_coord_t);
  
  af->buf = malloc(bufsz);
  if (verbose) {
    printf("anim: meta-data: ");
    ANIM_Print_metadata(stdout, af);
    fflush(stdout);
  }
  assert(af->buf);
  return af;
}

ANIM_File ANIM_Open(char * filename) {
  return open(filename, debugging);
}

size_t ANIM_Read_next(ANIM_File af) {
  size_t rv =
    fread(af->buf, (af->kind == ANIM_HEAT ? 1 : sizeof(ANIM_coord_t)),
	  af->frame_size, af->file);

  if (debugging) {
    printf("Read frame %zu from %s: %zu values read.\n",
	   af->framecount, af->filename, rv);
    fflush(stdout);
  }
  if (rv != 0) af->framecount++;
  return rv;
}

size_t ANIM_Read_frame(ANIM_File af, double * buf) {
  const int dim = af->dim;
  size_t rv = ANIM_Read_next(af);

  assert(buf);
  switch (af->kind) {
  case ANIM_HEAT: {
    const ANIM_byte * const color_buf = af->buf;
    const double min = af->ranges[dim].min, max = af->ranges[dim].max;

    for (size_t i=0; i<rv; i++)
      buf[i] = ANIM_byte_to_double(color_buf[i], min, max);
    break;
  }
  case ANIM_GRAPH: {
    const ANIM_coord_t * const coord_buf = af->buf;
    const double min = af->ranges[dim].min, max = af->ranges[dim].max;
    const int n = af->lengths[dim];
    
    for (size_t i=0; i<rv; i++)
      buf[i] = ANIM_coord_to_double(coord_buf[i], n, min, max);    
    break;
  }
  case ANIM_NBODY: {
    const ANIM_coord_t * coord_buf = af->buf;

    for (int i=0; i<dim; i++) {
      const double min = af->ranges[i].min, max = af->ranges[i].max;
      const int n = af->lengths[i];
      
      for (size_t j=i; j<rv; j += dim)
	buf[j] = ANIM_coord_to_double(coord_buf[j], n, min, max);
    }
    break;
  }
  default:
    assert(false);
  }
  return rv;
}

void ANIM_Close(ANIM_File af) {
  int err;
  
  if (debugging) {
    printf("Closing %s after writing %zu frames... ",
	   af->filename, af->framecount);
    fflush(stdout);
  }
  if (!af->read) { // writing, so write nframes to file
    err = fseek(af->file, ANIM_Get_nframes_offset(), SEEK_SET);
    assert(err == 0);
    size_t rv = fwrite(&af->framecount, sizeof(size_t), 1, af->file);
    assert(rv == 1);
  }
  err = fclose(af->file);
  if (err != 0) {
    fprintf(stderr,
	    "anim: warning: error occurred while closing file %s\n",
	    af->filename);
    fflush(stderr);
  }
  free(af->filename);
  free(af->buf);
  if (af->kind == ANIM_NBODY) {
    free(af->radii);
    free(af->colors);
    free(af->bcolors);
  }
  free(af);
  if (debugging) {
    printf("done.\n");
    fflush(stdout);
  }
}

void * ANIM_Get_buffer(ANIM_File af) {
  return af->buf;
}

int ANIM_Get_dim(ANIM_File af) {
  return af->dim;
}

int ANIM_Get_image_length(ANIM_File af, int index) {
  return af->lengths[index];
}

size_t ANIM_Get_frame_size(ANIM_File af) {
  return af->frame_size;
}

char * ANIM_Get_filename(ANIM_File af) {
  return af->filename;
}

size_t ANIM_Get_framecount(ANIM_File af) {
  return af->framecount;
}

ANIM_Kind ANIM_Get_kind(ANIM_File af) {
  return af->kind;
}

ANIM_range_t ANIM_Get_range(ANIM_File af, int index) {
  return af->ranges[index];
}

void ANIM_Print(char * filename) {
  ANIM_File af = open(filename, true);
  size_t size = ANIM_Get_frame_size(af);
  double * buf = malloc(size * sizeof(double));
  size_t framecount = 0, rv;

  assert(buf);
  while (true) {
    rv = ANIM_Read_frame(af, buf);
    if (rv == 0) break;
    printf("Frame %5zu: ", framecount); fflush(stdout);
    if (rv != size) {
      printf("Error: expected %zu doubles, but read only %zu\n",
	     size, rv);
      fflush(stdout);
    }
    for (size_t i=0; i<rv; i++) {
      printf("%7.2lf ", buf[i]);
      fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
    framecount++;
  }
  ANIM_Close(af);
  free(buf);
}

void ANIM_Seek_frame(ANIM_File af, size_t frame) {
  if (!af->read) {
    fprintf(stderr,
	    "anim: warning: ignoring seek request because writing file.\n");
    fflush(stderr);
    return;
  }
  if (frame > af->nframes) {
    fprintf(stderr, "anim: error: attempt to seek beyond end of file: %zu > %zu\n",
	    frame, af->nframes);
    fflush(stderr);
    exit(1);
  }

  long pos = (long)ANIM_File_size(af, frame);

  if (debugging) {
    printf("\nANIM: seeking frame %zu at byte offset %ld\n",
	   frame, pos);
    fflush(stdout);
  }

  int err = fseek(af->file, pos, SEEK_SET);

  if (err != 0) {
    fprintf(stderr, "anim: error: could not seek to file position %ld in %s\n",
	    pos, af->filename);
    fflush(stderr);
    exit(1);
  }
}

void ANIM_Status_update(FILE * stream, size_t nsteps, size_t step,
			int * statvar) {
  const int new_dots = nsteps > 0 ? 100L*step/nsteps : 100L;
  
  if (*statvar < new_dots) {
    do {
      fprintf(stream, "%c", ANIM_STATUS_BAR[*statvar]);
      (*statvar)++;
    } while (*statvar < new_dots);
    fflush(stream);
  }
}

double ANIM_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec/1000000.0;
}

double ** ANIM_allocate2d(int n, int m) {
  double * storage = malloc( n * m * sizeof(double) );
  double ** a = malloc( n * sizeof(double*) );
  
  assert(storage);
  assert(a);
  for (int i=0; i<n; i++) a[i] = & storage[ i * m ];
  return a;
}

void ANIM_free2d(double ** a) {
  free(a[0]); // frees storage
  free(a);    // frees a
}
