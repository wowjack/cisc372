/* anim2mp4.c :  Converts and ANIM file to an MPEG-4 movie.
 */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include "anim.h"
#include "convert.h"

// TODO: add same colormap option as in anim2gif
static void quit() {
  printf("Usage: anim2mp4 [OPTIONS] animfile                             \n \
Converts the ANIM file named animfile to an MPEG-4 movie.  If output     \n \
file name is not specified with -o: the output filename is obtained by   \n \
replacing the .anim suffix with .mp4, or appending .mp4 if animfile does \n \
not end with .anim.                                                      \n \
                                                                         \n \
Options:                                                                 \n \
  -o FILENAME      write the output to a file named FILENAME             \n \
  -keep            keep any intermediate files produced                  \n \
  -fps N           use N frames per second for the movie (default: 20)   \n \
  -color COLOR     use color map COLOR (redblue, gray)                   \n");
  exit(1);
}

int main(int argc, char * argv[]) {
  printf("anim2mp4: convert ANIM file to MP4 video\n");
  fflush(stdout);

  char * infilename = NULL, * outfilename = NULL;
  _Bool keep = false;
  int fps = 20;
  ANIM_Colormap cm = ANIM_REDBLUE;

  for (int i=1; i<argc; i++) {
    char * arg = argv[i];

    if (strcmp(arg, "-o") == 0) {
      if (++i >= argc) quit();
      outfilename = argv[i];
    } else if (strcmp(arg, "-color") == 0) {
      if (++i >= argc) quit();
      arg = argv[i];
      if (strcmp(arg, "gray") == 0)
	cm = ANIM_GRAY;
      else if (strcmp(arg, "redblue") == 0)
	cm = ANIM_REDBLUE;
      else quit();
    } else if (strcmp(arg, "-keep") == 0) {
      keep = true;
    } else if (strcmp(arg, "-fps") == 0) {
      if (++i >= argc) quit();
      fps = atoi(argv[i]);
    } else {
      if (infilename != NULL) quit();
      infilename = arg;
    }    
  }
  if (infilename == NULL) quit();
  if (outfilename == NULL)
    outfilename = ANIM_Make_filename(infilename, ".anim", ".mp4");
  if (fps < 1 || fps > 255) {
    fprintf(stderr, "anim: error: fps must be an integer in range 1..255");
    fflush(stderr);
    exit(1);
  }

  ANIM_File af = ANIM_Open(infilename);
  
  assert(af);
  ANIM_Make_mp4(af, cm, fps, outfilename, keep);
  ANIM_Close(af);
}
