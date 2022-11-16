#ifndef _ANIM_CONVERT
#define _ANIM_CONVERT
#include <stdio.h>
#include "anim.h"

typedef enum ANIM_Colormap {
  ANIM_REDBLUE, ANIM_GRAY
} ANIM_Colormap;

/* Creates a new filename based on an old filename using an
   appropriate suffix.  If the old filename ends with the old suffix,
   the new filename is formed by replacing the old suffix with the new
   suffix.  Otherwise, the new file name is formed by appending the
   new suffix to the old file name.  In any case, the new file name
   returned was allocated with malloc and can be freed with free. */
char * ANIM_Make_filename(const char * old_filename,
			  const char * old_suffix, const char * new_suffix);

void ANIM_Make_gif(ANIM_File af, ANIM_Colormap cm, FILE * out, char * out_fn);

void ANIM_gif2mp4(char * infilename, int fps, char * outfilename);

/* Convert the ANIM File to MPEG-4.
     af: the ANIM file handle, opened for reading, positioned at beginning
     cm: color map scheme
     fps: number of frames per second
     filename: name of the MPEG-4 file (usually ends with ".mp4")
     keep: keep intermediate files (such as animated GIFs)?
*/
void ANIM_Make_mp4(ANIM_File af, ANIM_Colormap cm, int fps, char * filename,
		   _Bool keep);

#endif
