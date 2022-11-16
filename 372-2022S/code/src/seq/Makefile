
stems = bounce cache diff1d diffuse1d diffusion1d hello \
        integral matmat nbody pascal perfect pi sat time
cleans = $(addsuffix .clean, $(stems))
tests = $(addsuffix .test, $(stems))

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
