.PHONY: clean run

CBLAS_DIR=/apps/x86-64/apps/spack_0.19.1/spack/opt/spack/linux-rocky8-zen3/gcc-11.3.0/netlib-lapack-3.10.1-vyp45bfucjjiiatmvtsi6xrnbymjuidi

main: main.cu
	nvcc -o main \
		-I. -I${CBLAS_DIR}/include \
		main.cu Timer.cpp \
		-L${CBLAS_DIR}/lib64 -lcblas \
		-Xlinker -rpath=${CBLAS_DIR}/lib64

test: main.cu
	nvcc -o main \
		-I. -I${CBLAS_DIR}/include \
		main.cu Timer.cpp \
		-L${CBLAS_DIR}/lib64 -lcblas \
		-Xlinker -rpath=${CBLAS_DIR}/lib64 \
		-lcublas

clean:
	rm -f main
