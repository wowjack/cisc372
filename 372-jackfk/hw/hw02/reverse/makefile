
build: reverse.c
	cc -o reverse.exec reverse.c

test: build
	#I'm not really sure how I would implement tests in a makefile
	#Maybe do some strange diff or cmp jank
	#I just make a bash script to check, hope it qualifies for extra credit
	bash test_script.sh

clean:
	rm -rf *.exec *.o
