ARCH = 75

all:
	python -m turingas.main -i lds32.sass -o lds32.cubin -arch $(ARCH)
	nvcc main.cu -lcuda -arch=sm_$(ARCH) -o test

clean:
	rm -f lds32.cubin test
