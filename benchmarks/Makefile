CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse

sddmm_cusparse.so: sddmm_cusparse.cu
	nvcc $(INC) $< -O3 -arch=native -Xcompiler -fPIC --shared -o $@ $(LIBS)

clean:
	rm -rf *.stdout *.stderr sddmm.pdf sddmm_cusparse.so __pycache__

clean-all: clean
	rm -f *-results.csv
