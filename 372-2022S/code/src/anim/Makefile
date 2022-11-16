ROOT = ../..
include $(ROOT)/common.mk

all: libanim.a anim2gif anim2mp4 anim2txt

test: # TODO

$(ROOT)/bin:
	[ -d $(ROOT)/bin ] || mkdir $(ROOT)/bin

$(ROOT)/lib:
	[ -d $(ROOT)/lib ] || mkdir $(ROOT)/lib

libanim.a: Makefile anim.o mpianim.o
	rm -f libanim.a
	ar rsv libanim.a anim.o mpianim.o
	rm -f $(ROOT)/lib/libanim.a
	cp libanim.a $(ROOT)/lib

anim.o: Makefile $(INC)/anim.h anim.c
	$(CCO) -DDEBUG -o anim.o anim.c

anim2txt: Makefile anim2txt.c $(INC)/anim.h $(INC)/anim_dev.h anim.o $(ROOT)/bin
	$(CCC) -o anim2txt anim2txt.c anim.o
	rm -f $(ROOT)/bin/anim2txt
	cp anim2txt $(ROOT)/bin

convert.o: Makefile $(INC)/anim.h convert.h convert.c
	$(CCO) -o convert.o convert.c

anim2gif: Makefile $(INC)/anim.h anim.o anim2gif.c convert.o $(ROOT)/bin
	$(CCC) -o anim2gif anim2gif.c anim.o convert.o -lgd
	rm -f $(ROOT)/bin/anim2gif
	cp anim2gif $(ROOT)/bin

anim2mp4: Makefile $(INC)/anim.h anim.o anim2mp4.c convert.o $(ROOT)/bin
	$(CCC) -o anim2mp4 anim2mp4.c anim.o convert.o -lgd
	rm -f $(ROOT)/bin/anim2mp4
	cp anim2mp4 $(ROOT)/bin

mpianim.o: Makefile $(INC)/mpianim.h mpianim.c
	$(MPICCO) -o mpianim.o mpianim.c

clean::
	rm -f libanim.a anim2mp4 anim2gif anim2txt
