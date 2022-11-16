/* convert: Convert an ANIM file to an animated GIF.
   Author: Stephen F. Siegel

   Some documentation from GD:

   void gdImageGifAnimBegin( gdImagePtr im, FILE * outFile, int GlobalCM,
     int Loops );

   gdImageSetPixel( gdImagePtr im, int x, int y, int color );

   void gdImageGifAnimAdd( gdImagePtr im, FILE *out, int LocalCM,
     int LeftOfs, int TopOfs, int Delay, int Disposal, gdImagePtr previm);

   Note: delay between frames appears to be 1/10 of a second, at least when 0
   is used for Delay.

   Note: y coordinates have to get flipped when making the GIF.
   That's because in ANIM, y increases as you go up, but in GIF, y
   increases as you go down.
*/
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <gd.h>
#include "convert.h"
#include "anim_dev.h"
// #define DEBUG

// experiment: different color schemes
//_Bool grayscale = true;
_Bool grayscale = false;

/* Creates and allocates a new color within the color map of the given
   image.  The function consumes an ANIM color, and converts it to a
   GD color and allocates.  The color is opaque (i.e., 0
   transparency). */
static int allocate_color(gdImagePtr image, ANIM_color_t color) {
  const int red = ANIM_Get_red(color), green = ANIM_Get_green(color),
    blue = ANIM_Get_blue(color);

#ifdef DEBUG
  printf("Allocating color %d,%d,%d\n", red, green, blue); fflush(stdout);
#endif

  const int result = gdImageColorAllocate(image, red, green, blue);

#ifdef DEBUG
  printf("Result is: %d\n", result); fflush(stdout);
#endif
  return result;
}

static void process_heat(ANIM_File af, ANIM_Colormap cm, FILE * out) {
  const size_t nframes = af->nframes;
  const size_t size = af->frame_size; // # scalars in one frame
  const int dim = af->dim;
  const int nx = ANIM_Get_image_length(af, 0);
  const int ny = dim == 1 ? 1 :  ANIM_Get_image_length(af, 1);
  const int pheight = dim == 1 ? 100 : 1;
  unsigned char * const buf = af->buf;
  gdImagePtr previm = NULL, im; // two consecutive animation images
  size_t count = 0; // number of frames processed
  int colors[ANIM_MAXCOLORS]; // colors we will use
  int stat = 0; // used for printing status bar

  if (dim >= 3) {
    fprintf(stderr,
	    "\nanim: for now, dimension of HEAT file must be 1 or 2, \
not %d\n", dim);
    exit(1);
  }
  while (true) {
    const size_t rv = ANIM_Read_next(af);
    
    if (rv == 0) break;
    if (rv != size) {
      fprintf(stderr,
	      "\nanim: error reading %s at frame %zu: expected \
%zu scalars, read %zu",
	      af->filename, af->framecount, size, rv);
      exit(1);
    }
    im = gdImageCreate(nx, ny*pheight);
    if (count == 0) {
      if (cm == ANIM_GRAY) {
	for (int j=0; j<ANIM_MAXCOLORS; j++)
	  colors[j] = gdImageColorAllocate(im, j, j, j);
      } else if (cm == ANIM_REDBLUE) {
	for (int j=0; j<ANIM_MAXCOLORS; j++)
	  colors[j] = gdImageColorAllocate(im, j, 0, ANIM_MAXCOLORS-j-1);
      } else {
	assert(0);
      }
      gdImageGifAnimBegin(im, out, 1, -1);
    } else {
      gdImagePaletteCopy(im, previm);
    }
    if (dim == 1) {
      for (int i=0; i<nx; i++)
	gdImageLine(im, i, 0, i, pheight-1, colors[buf[i]]);
    } else { // dim == 2
      for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	  gdImageSetPixel(im, i, ny - j - 1, colors[buf[i*ny + j]]);
    }
    if (count == 0) {
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, gdDisposalNone, NULL);
    } else {
      // Following is necessary due to bug in gd.
      // There must be at least one pixel difference between
      // two consecutive frames.  So I keep flipping one pixel.
      gdImageSetPixel(im, 0, 0, count%2);
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, gdDisposalNone, previm);
      gdImageDestroy(previm);
    }
    previm = im;
    count++;
    ANIM_Status_update(stdout, nframes, count, &stat);
  }
  if (previm != NULL) gdImageDestroy(previm);
  gdImageGifAnimEnd(out);
  printf("\n");
}

static void process_graph(ANIM_File af, FILE * out) {
  const size_t nframes = af->nframes;
  const size_t size = af->frame_size; // # scalars in one frame
  const int dim = af->dim;
  const int nx = ANIM_Get_image_length(af, 0);
  const int ny = ANIM_Get_image_length(af, 1);
  ANIM_coord_t * const coord_buf = (ANIM_coord_t*)af->buf;
  gdImagePtr previm = NULL, im; // two consecutive animation images
  size_t count = 0; // number of frames processed
  int white, black, trans;  // colors used
  int stat = 0; // used for printing status bar

  // TODO: handle 2-d
  if (dim != 1) {
    fprintf(stderr,
	    "\nanim: for now, dimension of GRAPH file must be 1, not %d\n",
	    dim);
    exit(1);
  }
  assert(dim == 1);
  while (true) {
    const size_t rv = ANIM_Read_next(af);
    
    if (rv == 0) break;
    if (rv != size) {
      fprintf(stderr,
	      "\nanim: error reading %s at frame %zu: expected \
%zu scalars, read %zu", af->filename, af->framecount, size, rv);
      exit(1);
    }
#ifdef DEBUG
    fprintf(stderr, "Starting graph image creation\n");
    fflush(stderr);
#endif
    im = gdImageCreate(nx, ny);
    if (count == 0) {
      white = gdImageColorAllocate(im, 255, 255, 255); // white
      /* Allocate drawing color */
      black = gdImageColorAllocate(im, 0, 0, 0);
      /* Allocate transparent color for animation compression */
      trans = gdImageColorAllocate(im, 1, 1, 1);
      gdImageGifAnimBegin(im, out, 1, -1);
    } else {
      /* Allocate background to make it white */
      (void)gdImageColorAllocate(im, 255, 255, 255);
      gdImagePaletteCopy(im, previm);
    }
    /* Need to make sure at least one pixel differs from previous
       slide or GD fails due to defect... */
#ifdef DEBUG
    fprintf(stderr, "Setting pixel %d, %d\n", (int)(count%nx), ny-1);
    fflush(stderr);
#endif
    gdImageSetPixel(im, 0, 0, (count%2 ? white : black));

    ANIM_coord_t y_prev = ny - coord_buf[0] - 1;
    
    for (int i=1; i<nx; i++) {
      const ANIM_coord_t y = ny - coord_buf[i] - 1;

#ifdef DEBUG
      fprintf(stderr, "Drawing line %d, %d, %d, %d\n", i-1, y_prev, i, y);
      fflush(stderr);
#endif
      gdImageLine(im, i-1, y_prev, i, y, black);
      y_prev = y;
    }
    if (count == 0) {
#ifdef DEBUG
      fprintf(stderr, "Adding image %zu...\n", count);
      fflush(stderr);
#endif
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, 1, NULL);
    } else {
#ifdef DEBUG
      fprintf(stderr, "Making color transparent...\n");
      fflush(stderr);
#endif
      gdImageColorTransparent(im, trans);
#ifdef DEBUG
      fprintf(stderr, "Adding image %zu...\n", count);
      fflush(stderr);
#endif
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, 1,  previm);
#ifdef DEBUG
      fprintf(stderr, "Destorying previous image...\n");
      fflush(stderr);
#endif
      gdImageDestroy(previm);
    }
    previm = im;
    count++;
    ANIM_Status_update(stdout, nframes, count, &stat);
  }
  if (previm != NULL) gdImageDestroy(previm);
  gdImageGifAnimEnd(out);
  printf("\n");
}

static void process_nbody(ANIM_File af, FILE * out) {
  const size_t nframes = af->nframes;
  const size_t size = af->frame_size; // # scalars in one frame
  const int dim = af->dim;
  const int nx = ANIM_Get_image_length(af, 0);
  const int ny = dim == 1 ? 1 :  ANIM_Get_image_length(af, 1);
  //const int pheight = dim == 1 ? 100 : 1; // TODO: use this in 1d
  unsigned char * const buf = af->buf;
  gdImagePtr previm = NULL, im; // two consecutive animation images
  size_t count = 0; // number of frames processed
  const int nbodies = af->nbodies, ncolors = af->ncolors;
  const int * const radii = af->radii, * const bcolors = af->bcolors;
  ANIM_coord_t * const coord_buf = (ANIM_coord_t*)buf;
  int colors[ncolors];
  int stat = 0;
  
  if (ncolors > ANIM_MAXCOLORS) {
    fprintf(stderr, "\nanim: maximum number of colors (%d) exceeded: %d\n",
	    ANIM_MAXCOLORS, ncolors);
    fflush(stderr);
    exit(1);
  }
  if (dim >= 3) {
    fprintf(stderr,
	    "\nanim: for now, dimension of NBODY simulation must be \
1 or 2, not %d\n", dim);
    exit(1);
  }
  while (true) {
    const size_t rv = ANIM_Read_next(af);
    
    if (rv == 0) break;
    if (rv != size) {
      fprintf(stderr,
	      "\nanim: error reading %s at frame %zu: expected \
%zu scalars, read %zu",
	      af->filename, af->framecount, size, rv);
      exit(1);
    }
    im = gdImageCreate(nx, ny);
    if (count == 0) {
      gdImageColorAllocate(im, 0, 0, 0);  /* black background */
      for (int i=0; i<ncolors; i++) {
#ifdef DEBUG
	printf("Color = %d\n", af->colors[i]);
#endif
	colors[i] = allocate_color(im, af->colors[i]);
      }
      gdImageGifAnimBegin(im, out, 1, -1);
    } else {
      gdImagePaletteCopy(im, previm);
    }
    // dimension 1, 2, or 3
    if (dim == 1) {
      // TODO: draw as solid square of given radius resting on ground
      fprintf(stderr, "\nanim: NBODY dimension 1 not yet supported\n");
      exit(1);
    } else if (dim == 2) { // solid circle of given radius
      for (int i=0, j=0; i<nbodies; i++) {
	const int posx = coord_buf[j++], posy = ny - coord_buf[j++] - 1;
	const int radius = radii[i], colorIdx = bcolors[i];

#ifdef DEBUG
	printf("posx=%d, posy=%d, radius=%d, color=%d...",
	       posx, posy, radius, colors[colorIdx]);
	fflush(stdout);
#endif
	gdImageFilledEllipse(im, posx, posy, radius, radius, colors[colorIdx]);
#ifdef DEBUG
	printf("done.\n"); fflush(stdout);
#endif
      }
    } else if (dim == 3) {
      // TODO: draw as circle but shrink or fade to reflect z-distance?
      assert(false); // not yet implemented
    } else
      assert(false); // can't happen
    /* to ensure frame is different from previous, change one pixel.
       workaround for bug in GD */
    gdImageSetPixel(im, 0, 0, count%2);
    if (count == 0) {
#ifdef DEBUG
      printf("Adding image..."); fflush(stdout);
#endif
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, gdDisposalNone, NULL);
#ifdef DEBUG
      printf("done.\n"); fflush(stdout);
#endif
    } else {
#ifdef DEBUG
      printf("Adding image..."); fflush(stdout);
#endif
      gdImageGifAnimAdd(im, out, 0, 0, 0, 0, gdDisposalNone, previm);
#ifdef DEBUG
      printf("done.\n"); fflush(stdout);
#endif
      gdImageDestroy(previm);
    }
    previm = im;
    count++;
    ANIM_Status_update(stdout, nframes, count, &stat);
  }
  if (previm != NULL) gdImageDestroy(previm);
  gdImageGifAnimEnd(out);
  printf("\n");
}

void ANIM_Make_gif(ANIM_File af, ANIM_Colormap cm, FILE * out, char * out_fn) {
  printf("anim: converting %s to animated GIF %s:\n",
	 af->filename, out_fn);
  fflush(stdout);
  switch (af->kind) {
  case ANIM_HEAT:
    process_heat(af, cm, out);
    break;
  case ANIM_GRAPH:
    process_graph(af, out);
    break;
  case ANIM_NBODY:
    process_nbody(af, out);
    break;
  default:
    assert(false);
  }
  printf("anim: construction of %s complete.\n", out_fn);
}

/* Determines whether the string s ends with the string suffix.
   If so, the index into s of the first character of the suffix
   is returned.  Otherwise, -1 is returned */
static int ANIM_suffix(const char * s, const char * suffix) {
  const int n = strlen(s) - strlen(suffix);

  if (n < 0) return -1;
  for (const char * p = s+n, * q = suffix; *p; p++, q++) {
    if (*p != *q) return -1;
  }
  return n;
}

/* Creates a new filename based on an old filename using an
   appropriate suffix.  If the old filename ends with the old suffix,
   the new filename is formed by replacing the old suffix with the new
   suffix.  Otherwise, the new file name is formed by appending the
   new suffix to the old file name.  In any case, the new file name
   returned was allocated with malloc and can be freed with free. */
char * ANIM_Make_filename(const char * old_filename,
			  const char * old_suffix, const char * new_suffix) {
  int idx = ANIM_suffix(old_filename, old_suffix);
  char * result;

  if (idx < 0) { // old_filename does not end with old_suffix
    result = malloc(strlen(old_filename) + strlen(new_suffix) + 2);
    sprintf(result, "%s%s", old_filename, new_suffix);
  } else { // old_filename ends with old_suffix
    result = malloc(idx + strlen(new_suffix) + 1);
    memcpy(result, old_filename, idx);
    sprintf(result + idx, "%s", new_suffix);
  }
  return result;
}

// TODO: add a debug option to enable these...

// loglevels: "quiet" "panic" "fatal" "error" "warning" "info" "verbose" "debug" "trace"

#define FORMAT "ffmpeg -itsscale %.10lf -i %s -r %d -y \
-loglevel \"warning\" -movflags faststart -pix_fmt yuv420p \
-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s"

void ANIM_gif2mp4(char * infilename, int fps, char * outfilename) {
  assert(1<= fps && fps<=255);
  assert(infilename);
  assert(outfilename);
  // the animated GIF has 10 fps.  So its time period must be scaled.
  // E.g., to get 60 fps, scale by 10/60.
  double scale = 10.0/fps;
  char cmd[strlen(FORMAT) + 20 + strlen(infilename) + 3 + strlen(outfilename)];

  sprintf(cmd, FORMAT, scale, infilename, fps, outfilename);
  printf("anim: converting animated GIF %s to MPEG-4 %s at %d fps...",
	 infilename, outfilename, fps);
  //printf("%s\n", cmd);
  fflush(stdout);
  
  int result = system(cmd);
  
  if (result != 0) {
    fprintf(stderr, "\nanim: error: ffmpeg returned non-0 exit code %d\n", result);
    fflush(stderr);
    exit(1);
  }
  printf("done.\n");
  fflush(stdout);
}

void ANIM_Make_mp4(ANIM_File af, ANIM_Colormap cm, int fps,
		   char * filename, _Bool keep) {
  char * gif_name = ANIM_Make_filename(af->filename, ".anim", ".gif");
  // TODO: Problem: what if gif_name equals filename?  Change it or report.
  FILE * gif_stream = fopen(gif_name, "wb");

  if (!gif_stream) {
    fprintf(stderr, "anim: failed to open file %s for writing\n",
	    gif_name);
    fflush(stderr);
    exit(1);
  }
  ANIM_Make_gif(af, cm, gif_stream, gif_name);
  if (fclose(gif_stream)) {
    fprintf(stderr, "anim: error: could not close file %s\n",
	    gif_name);
    fflush(stderr);
    exit(1);
  }
  ANIM_gif2mp4(gif_name, fps, filename);
  if (!keep && remove(gif_name)) {
    fprintf(stderr, "anim: warning: could not delete temporary file %s\n",
	    gif_name);
    fflush(stderr);
  }
  free(gif_name);
}
