ARCH := $(shell python cuda_check.py | grep "Compute Capability:" | cut -d: -f 2 | xargs)
all: stencil matmult
stencil: stencil.cu
	nvcc -arch $(ARCH) stencil.cu -O3 -o stencil
matmult: mm.cu
	nvcc -arch $(ARCH) mm.cu -I /uufs/chpc.utah.edu/sys/installdir/cuda/10.1.168/samples/common/inc -o matmult
clean:
	rm -f matmult stencil
