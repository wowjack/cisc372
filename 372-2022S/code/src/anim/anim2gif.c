/* anim2gif.c :  Converts an ANIM file to an animated GIF.
 */
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "anim.h"
#include "convert.h"

// TODO: add colormap option: -color redblue (red-blue), gray (grayscale)

static void quit() {
  printf("Usage: anim2gif [OPTIONS] animfile                             \n \
Converts the ANIM file named animfile to an animated GIF.  If output     \n \
file name is not specified with -o: the output filename is obtained by   \n \
replacing the .anim suffix with .gif, or appending .gif if animfile does \n \
not end with .anim.                                                      \n \
                                                                         \n \
Options:                                                                 \n \
  -o FILENAME     write the output to file named FILENAME                \n \
  -color COLOR    use color map COLOR (redblue, gray)                    \n \
");
  exit(1);
}

int main(int argc, char * argv[]) {
  char * infilename = NULL, * outfilename = NULL;
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
    } else {
      if (infilename != NULL) quit();
      infilename = arg;
    }    
  }
  if (infilename == NULL) quit();
  if (outfilename == NULL)
    outfilename = ANIM_Make_filename(infilename, ".anim", ".gif");

  ANIM_File af = ANIM_Open(infilename);
  
  assert(af);

  FILE * out = fopen(outfilename, "wb");
  
  if (out == NULL) {
    fprintf(stderr, "anim: could not open file %s for writing\n",
	    outfilename);
    exit(1);
  }
  ANIM_Make_gif(af, cm, out, outfilename);
  fclose(out);
  ANIM_Close(af);
}
