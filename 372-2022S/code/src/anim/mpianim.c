/* Filename : mpianimc.c
   Author   : Stephen F. Siegel
   Created  : 23-jun-2020
   Modified : 23-jun-2020

   Implementation of mpianim.h.  We use MPI's I/O functions for file
   manipulation.  This should provide good performance on even the
   largest clusters.

   This file is to be linked with anim.c.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "mpianim.h"
#include "anim_dev.h"

/* The MPI Datatypes corresponding to our C types ANIM_coord_t and
   ANIM_color_t */
#define ANIM_coord_datatype MPI_UNSIGNED_SHORT
#define ANIM_color_datatype MPI_UNSIGNED

/* Are we debugging?  Linked to the debugging global variables in
   anim.c */
extern _Bool debugging;

struct MPIANIM_File_s {
  
  /* What kind of animation is this? */
  ANIM_Kind kind;
  
  /* The MPI communicator that will be used for all communication in
     this library.  A duplicate of the communicator passed in at
     creation */
  MPI_Comm comm;

  /* MPI rank of this process in comm */
  int rank;

  /* The name of the file */
  char * filename;
  
  /* The MPI file handle for the file that is being constructed. */
  MPI_File file;

  /* The MPI_Datatype corresponding to a single scalar element of
     compressed data.  For HEAT, it is MPI_BYTE; for GRAPH and NBODY,
     MPI_UNSIGNED_SHORT.*/
  MPI_Datatype etype;
  
  /* The MPI datatype corresponding to the part of the file to which
     this process will write in one frame. */
  MPI_Datatype filetype;

  /* The MPI datatype corresponding to C's size_t. */
  MPI_Datatype size_datatype;

  /* Are we reading (not writing) this file? */
  bool read;
  
  /* The dimension of the spatial domain of the function being
     depicted: 1, 2, or 3. */
  int dim;
  
  /* Length of each dimension of the image, in pixels. For HEAT and
     NBODY, this is an array of length dim.  For GRAPH, length
     dim+1. */
  int lengths[3];
  
  /* The intervals defining the domain and range of the function.  For
     HEAT and GRAPH, this is an array of length dim+1.  For NBODY, the
     length is dim. */
  ANIM_range_t ranges[4];
  
  /* The total number of scalar elements in one frame of the
     animation.  For HEAT and GRAPH, this is lengths[0] * ... *
     lengths[dim-1].  For NBODY, it is nbodies*dim. */
  size_t frame_size;
  
  /* The number of scalar elements in one animation frame for which
     this process is responsible */
  size_t local_frame_size;

  /* The number of frames in the final animation, set at very end of
     writing, when file is closed. */
  size_t nframes;
  
  /* The number of frames written or read so far */
  size_t framecount;
  
  /* A buffer holding compressed data for one frame.  This buffer can
     hold local_frame_size scalars.  For HEAT, a scalar is one byte.
     For GRAPH and NBODY, a scalar is one unsigned short. */
  void * buf;

  // The following fields are used for NBODY animations only...
  
  /* The global number of bodies.  Must be the same on every proc.
     (used for NBODY only) */
  int nbodies;
  
  /* The number of bodies stored on this proc only.  Used for NBODY
     only. */
  int blocksize;
  
  /* The radius of the sphere used to represent each body, in pixels.
     Array of length blocksize.  Used for NBODY only */
  int * radii;
  
  /* The total number of colors that will be used in this NBODY
     animation. Must be the same on every proc.  Used for NBODY
     only.  */
  int ncolors;
  
  /* The colors.  Array of length ncolors.  Must be the same on every
     proc.  Used for NBODY only.  */
  ANIM_color_t * colors;
  
  /* Color of each body owned by this proc.  Array of length
     blocksize. */
  int * bcolors;
};

/* Prints the data in af.  To be used for debugging */
static void print_metadata(MPIANIM_File af) {
  int dim = af->dim;
  
  printf("%s (kind = %s, rank = %d, dim = %d):",
	 af->filename, ANIM_kind_to_str(af->kind), af->rank, dim);
  printf("\n  open for %sing", (af->read ? "read" : "writ"));
  printf("\n  lengths = ");
  ANIM_Print_int_array(stdout, (af->kind == ANIM_GRAPH ? dim+1 : dim), af->lengths );
  printf("\n  ranges = ");
  ANIM_Print_range_array(stdout, (af->kind == ANIM_NBODY ? dim : dim+1), af->ranges );
  printf("\n  frame_size = %zu", af->frame_size);
  printf("\n  local_frame_size = %zu", af->local_frame_size);
  printf("\n  nframes = %zu", af->nframes);
  printf("\n  frame_count = %zu", af->framecount);
  printf("\n  buf = %p", af->buf);
  if (af->kind == ANIM_NBODY) {
    printf("\n  nbodies = %d", af->nbodies);
    // TODO: finish me...
  }
  printf("\n");
}

static MPIANIM_File
create_common(ANIM_Kind kind, MPI_Comm comm, char * filename,
	      MPI_Datatype etype, MPI_Datatype filetype, 
	      int dim, int nlengths, int * lengths,
	      int nranges, ANIM_range_t * ranges, size_t frame_size,
	      size_t local_frame_size) {
  MPIANIM_File af = malloc(sizeof(struct MPIANIM_File_s));
  int err;
  
  assert(af);
  af->kind = kind;
  err = MPI_Comm_dup(comm, &af->comm);
  assert(err == 0);
  err = MPI_Comm_rank(af->comm, &af->rank);
  assert(err == 0);  
  assert(filename);
  af->filename = malloc(strlen(filename)+1);
  assert(af->filename);
  strcpy(af->filename, filename);
  err =  MPI_File_open(af->comm, filename,
		       MPI_MODE_CREATE | MPI_MODE_WRONLY,
		       MPI_INFO_NULL, &af->file);
  assert(err == 0);
  // in case the file already existed, truncate it to 0:
  err = MPI_File_set_size(af->file, 0);
  assert(err == 0);
  err = MPI_File_set_view(af->file, 0, MPI_BYTE, MPI_BYTE,
			  "native", MPI_INFO_NULL);
  assert(err == 0);
  af->etype = etype;
  af->filetype = filetype;

  const size_t sizesz = sizeof(size_t);

  if (sizesz == sizeof(unsigned))
    af->size_datatype = MPI_UNSIGNED;
  else if (sizesz == sizeof(unsigned long))
    af->size_datatype = MPI_UNSIGNED_LONG;
  else if (sizesz == sizeof(unsigned long long))
    af->size_datatype = MPI_UNSIGNED_LONG_LONG;
  else {
    fprintf(stderr, "I have no idea what the type size_t is.\n");
    fflush(stderr);
    exit(1);
  }
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
  af->frame_size = frame_size;
  af->local_frame_size = local_frame_size;
  af->nframes = 0;
  af->framecount = 0;

  if (local_frame_size == 0) {
    af->buf = NULL;
  } else {
    const size_t bufsz = kind == ANIM_HEAT ? local_frame_size :
      local_frame_size * sizeof(ANIM_coord_t);

    af->buf = malloc(bufsz);
    assert(af->buf);
  }
  // kind NBODY can overwrite this part
  af->nbodies = 0;
  af->blocksize = 0;
  af->radii = NULL;
  af->ncolors = 0;
  af->colors = NULL;
  af->bcolors = NULL;
  // end NBODY part
  return af;
}

/* Writes the common part of the file header to the file.  When this
   function returns, the file view, for all processes, will be at the
   beginning of the file with both etype and filetype MPI_BYTE.  The
   shared file pointer will point to the end of the file.  Individual
   file pointers will be 0.   Collective operation. */
static void write_header_common(MPIANIM_File af) {
  if (debugging) {
    int nprocs;

    MPI_Comm_size(af->comm, &nprocs);
    for (int i=0; i<nprocs; i++) {
      if (i == af->rank) {
	print_metadata(af);
	fflush(stdout);
      }
      MPI_Barrier(af->comm);
    }
  }
  if (af->rank == 0) {
    int kind_code = ANIM_kind_to_int(af->kind), count;
    MPI_Status status;

    // MPI_File_write_shared updates the shared file pointer
    int err = MPI_File_write_shared(af->file, &kind_code, 1, MPI_INT, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_INT, &count);
    assert(err == 0);
    assert(count == 1);
    err = MPI_File_write_shared(af->file, &af->dim, 1, MPI_INT, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_INT, &count);
    assert(count == 1);
    err = MPI_File_write_shared(af->file, af->lengths, 3, MPI_INT, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_INT, &count);
    assert(count == 3);
    err = MPI_File_write_shared(af->file, (double*)af->ranges, 8,
				MPI_DOUBLE, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_DOUBLE, &count);
    assert(count == 8);
    err = MPI_File_write_shared(af->file, &af->nframes, 1, af->size_datatype,
				&status);
    assert(err == 0);
    err = MPI_Get_count(&status, af->size_datatype, &count);
    assert(count == 1);    
    err = MPI_File_write_shared(af->file, &af->frame_size, 1, af->size_datatype,
				&status);
    assert(err == 0);
    err = MPI_Get_count(&status, af->size_datatype, &count);
    assert(err == 0);
    assert(count == 1);
    if (debugging) {
      printf("File opened and header written.\n");
      fflush(stdout);
    }
  }
}

/* Takes current position of the shared file pointer and uses this to
   set the new view for all processes.  The view's etype and filetype
   will be set to the corresponding fields of af.  All local and the
   shared file pointers will be set to 0.  Collective operation. */
static void end_header(MPIANIM_File af) {
  MPI_Offset disp, offset;

  MPI_Barrier(af->comm);
  MPI_File_get_position_shared(af->file, &offset);
  MPI_File_get_byte_offset(af->file, offset, &disp);
  int err = MPI_File_set_view(af->file, disp, af->etype, af->filetype,
			      "native", MPI_INFO_NULL);

  if (debugging) {
    printf("Proc %d: new view with disp = %lu\n", af->rank, (unsigned long)disp);
    fflush(stdout);
  }
  MPI_Barrier(af->comm);  
  assert(err == 0);
}

MPIANIM_File MPIANIM_Create_heat(int dim,
				 int lengths[dim],
				 ANIM_range_t ranges[dim+1],
				 int sublengths[dim],
				 int starts[dim],
				 char * filename,
				 MPI_Comm comm) {
  // all procs must agree on dim, lengths, ranges, filename, comm
  // sublengths and starts must partition domain --- can that be checked?
  size_t frame_size = 1, local_frame_size = 1;
  MPI_Datatype etype = MPI_UNSIGNED_CHAR, filetype;
  int err;

  for (int i=0; i<dim; i++) {
    frame_size *= lengths[i];
    local_frame_size *= sublengths[i];
  }
  if (local_frame_size == 0) {
    filetype = etype; // this proc will never write data
  } else {
    err = MPI_Type_create_subarray(dim, lengths, sublengths, starts,
				   MPI_ORDER_C, etype, &filetype);
    assert(err == 0);
    err = MPI_Type_commit(&filetype);
    assert(err == 0);
  }
  
  MPIANIM_File af =
    create_common(ANIM_HEAT, MPI_COMM_WORLD, filename, etype, filetype,
		  dim, dim, lengths, dim+1, ranges,
		  frame_size, local_frame_size);

  write_header_common(af);
  end_header(af);
  return af;
}

MPIANIM_File MPIANIM_Create_graph(int dim,
				  int lengths[dim+1],
				  ANIM_range_t ranges[dim+1], 
				  int sublengths[dim],
				  int starts[dim],
				  char * filename,
				  MPI_Comm comm) {
  size_t frame_size = 1, local_frame_size = 1;
  MPI_Datatype etype = ANIM_coord_datatype, filetype;
  int err;

  for (int i=0; i<dim; i++) {
    frame_size *= lengths[i];
    local_frame_size *= sublengths[i];
  }
  if (local_frame_size == 0) {
    filetype = etype;
  } else {
    err = MPI_Type_create_subarray(dim, lengths, sublengths, starts,
				   MPI_ORDER_C, etype, &filetype);
    assert(err == 0);
    err = MPI_Type_commit(&filetype);
    assert(err == 0);
  }

  MPIANIM_File af =
    create_common(ANIM_GRAPH, MPI_COMM_WORLD, filename, etype, filetype,
		  dim, dim+1, lengths, dim+1, ranges,
		  frame_size, local_frame_size);

  write_header_common(af);
  end_header(af);
  return af;
}

MPIANIM_File MPIANIM_Create_nbody(int dim,
				  int lengths[dim],
				  ANIM_range_t ranges[dim],
				  int ncolors, 
				  ANIM_color_t colors[ncolors], 
				  int nbodies,
				  int blocksize,
				  int first,
				  int radii[blocksize],
				  int bcolors[blocksize],
				  char * filename,
				  MPI_Comm comm) {
  MPI_Status status;
  MPI_Datatype etype = ANIM_coord_datatype, filetype;
  int err, count;

  assert(nbodies >= 1);
  if (blocksize == 0) {
    filetype = etype; // this proc will never write data
  } else {
    err =
      MPI_Type_create_subarray(1, (int[]){nbodies*dim}, (int[]){blocksize*dim},
			       (int[]){first*dim}, MPI_ORDER_C, etype, &filetype);
    assert(err == 0);
    err = MPI_Type_commit(&filetype);
    assert(err == 0);
  }
  
  MPIANIM_File af =
    create_common(ANIM_NBODY, MPI_COMM_WORLD, filename, etype, filetype,
		  dim, dim, lengths, dim, ranges,
		  nbodies*dim, blocksize*dim);

  af->nbodies = nbodies;
  af->radii = malloc(blocksize * sizeof(int));
  assert(af->radii);
  for (int i=0; i<blocksize; i++) af->radii[i] = radii[i];
  assert(ncolors >= 1);
  af->ncolors = ncolors;
  af->colors = malloc(ncolors * sizeof(ANIM_color_t));
  assert(af->colors);
  for (int i=0; i<ncolors; i++) af->colors[i] = colors[i];
  af->bcolors = malloc(blocksize * sizeof(int));
  for (int i=0; i<blocksize; i++) af->bcolors[i] = bcolors[i];
  write_header_common(af);
  MPI_Barrier(af->comm);
  if (af->rank == 0) {
    err = MPI_File_write_shared(af->file, &nbodies, 1, MPI_INT, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_INT, &count);
    assert(count == 1);
  }
  MPI_Barrier(af->comm);
  err = MPI_File_write_ordered(af->file, radii, blocksize, MPI_INT, &status);
  assert(err == 0);
  MPI_Get_count(&status, MPI_INT, &count);
  assert(count == blocksize);
  MPI_Barrier(af->comm);
  if (af->rank == 0) {
    err = MPI_File_write_shared(af->file, &ncolors, 1, MPI_INT, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, MPI_INT, &count);
    assert(count == 1);
    err = MPI_File_write_shared(af->file, colors, ncolors,
				ANIM_color_datatype, &status);
    assert(err == 0);
    err = MPI_Get_count(&status, ANIM_color_datatype, &count);
    assert(count == ncolors);
  }
  MPI_Barrier(af->comm);
  err = MPI_File_write_ordered(af->file, bcolors, blocksize, MPI_INT, &status);
  assert(err == 0);
  err = MPI_Get_count(&status, MPI_INT, &count);
  assert(count == blocksize);
  end_header(af);
  return af;
}

void MPIANIM_Set_nframes(MPIANIM_File af, size_t nframes) {
  size_t file_size;
  
  switch (af->kind) {
  case ANIM_HEAT:
    file_size = ANIM_Heat_file_size(af->dim, af->lengths, nframes);
    break;
  case ANIM_GRAPH:
    file_size = ANIM_Graph_file_size(af->dim, af->lengths, nframes);
    break;
  case ANIM_NBODY:
    file_size =
      ANIM_Nbody_file_size(af->dim, af->nbodies, af->ncolors, nframes);
    break;
  default:
    assert(0); // unreachable
    exit(1);
  }
  int err = MPI_File_set_size(af->file, (MPI_Offset)file_size);
  if (err != 0) {
    fprintf(stderr, "mpianim: Unable to resize file %s to %zu bytes\n",
	    af->filename, file_size);
    exit(1);
  }
}

static void write_frame_heat(MPIANIM_File af, double * buf,
			     MPI_Status * status) {
  int dim = af->dim;
  size_t size = af->local_frame_size;
  unsigned char * color_buf = af->buf;
  const double min = af->ranges[dim].min, max = af->ranges[dim].max;

  if (debugging) {
    printf("Proc %d: writing frame %zu to %s:\n", af->rank,
	   af->framecount, af->filename);
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
  int err = MPI_File_write_all(af->file, color_buf, size,
			       MPI_UNSIGNED_CHAR, status);
  assert(err == 0);
}

static void write_frame_graph(MPIANIM_File af, double * buf,
			      MPI_Status * status) {
  const int dim = af->dim;
  const size_t size = af->local_frame_size;
  ANIM_coord_t * const coord_buf = af->buf;
  const double min = af->ranges[dim].min, max = af->ranges[dim].max;
  const int n = af->lengths[dim];

  for (size_t i=0; i<size; i++)
    coord_buf[i] = ANIM_double_to_coord(buf[i], n, min, max);
  int err = MPI_File_write_all(af->file, coord_buf, size,
			       ANIM_coord_datatype, status);
  assert(err == 0);
}

static void write_frame_nbody(MPIANIM_File af, double * buf,
			      MPI_Status * status) {
  int dim = af->dim;
  size_t size = af->local_frame_size;
  ANIM_coord_t * const coord_buf = af->buf;

  for (int i=0; i<dim; i++) {
    const double min = af->ranges[i].min, max = af->ranges[i].max;
    const int n = af->lengths[i];
      
    for (size_t j=i; j<size; j += dim)
	coord_buf[j] = ANIM_double_to_coord(buf[j], n, min, max);
  }
#ifdef DEBUG
  printf("Rank %d: writing frame: ", af->rank);
  for (size_t i=0; i<size; i++) printf("%u ", coord_buf[i]);
  printf("\n");
  fflush(stdout);
#endif
  
  int err = MPI_File_write_all(af->file, coord_buf, size,
			       ANIM_coord_datatype, status);
  assert(err == 0);
}

void MPIANIM_Write_frame(MPIANIM_File af, double * buf,
			 MPI_Status * status) {
  switch(af->kind) {
  case ANIM_HEAT:
    write_frame_heat(af, buf, status);
    break;
  case ANIM_GRAPH:
    write_frame_graph(af, buf, status);
    break;
  case ANIM_NBODY:
    write_frame_nbody(af, buf, status);
    break;
  default:
    assert(false);
  }
  af->framecount++;
}

void MPIANIM_Close(MPIANIM_File af) {
  int err;
  
  if (debugging) {
    printf("Closing %s.\n", af->filename);
    fflush(stdout);
  }
  if (!af->read) {
    err = MPI_File_set_view(af->file, 0, MPI_BYTE, MPI_BYTE,
			    "native", MPI_INFO_NULL);
    assert(err == 0);
    if (af->rank == 0) {
      MPI_Offset offset = (MPI_Offset)ANIM_Get_nframes_offset(); // was long
      MPI_Status status;
      int count;

      err = MPI_File_write_at(af->file, offset, &af->framecount, 1,
			      af->size_datatype, &status);
      assert(err == 0);
      err = MPI_Get_count(&status, af->size_datatype, &count);
      assert(err == 0);
      assert(count == 1);
    }
  }
  err = MPI_File_close(&af->file);
  assert(err == 0);
  if (af->filetype != af->etype) {
    err = MPI_Type_free(&af->filetype);
    assert(err == 0);
  }
  free(af->filename);
  if (af->local_frame_size > 0) free(af->buf);
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
