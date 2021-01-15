ARCH = 80

all:
	python -m turingas.main -i lds32.sass -o lds32.cubin -arch $(ARCH)
	nvcc main.cu -lcuda -arch=sm_$(ARCH) -o test

basic_LDG:
	python -m turingas.main -i basic_LDG.sass -o basic_LDG.cubin -arch $(ARCH)
	nvcc basic_LDG.cu -lcuda -arch=sm_$(ARCH) -o basic_LDG

LDG_profile:
	python -m turingas.main -i LDG_profile.sass -o LDG_profile.cubin -arch $(ARCH)
	nvcc  -g -G -O0 LDG_profile.cu -lcuda -arch=sm_$(ARCH) -o LDG_profile

clean:
	rm -f basic_LDG.cubin LDG_profile.cubin basic_LDG LDG_profile
