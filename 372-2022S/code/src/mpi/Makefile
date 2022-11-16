
stems = args datatype collective diffuse1d diffusion1d distribute \
        hello matmat mpianim pascal p2p pi sat
tests = $(addsuffix .test, $(stems))
cleans = $(addsuffix .clean, $(stems))

all: $(stems)

test: $(tests)

clean: $(cleans)
	rm -f *~ *.tmp *.exec *.o a.out

$(stems): %:
	$(MAKE) -C $* all

$(tests): %.test:
	$(MAKE) -C $* test

$(cleans): %.clean:
	$(MAKE) -C $* clean

.PHONY: all test clean $(stems) $(tests) $(cleans)
