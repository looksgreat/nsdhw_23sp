CXX = g++
CXXFLAGS = -std=c++11 -O3 -shared
HEADERS = -I. #-I/DATA/lllll/.local/lib/python3.6/site-packages/pybind11/include
PIBINDFLAG = -fPIC `python3 -m pybind11 --includes` `python3-config --includes --ldflags`
PY_OBJ = _matrix.so
MKLROOT=/usr/lib/x86_64-linux-gnu
LDFLAGS=${MKLROOT}/libmkl_def.so \
		${MKLROOT}/libmkl_avx2.so \
		${MKLROOT}/libmkl_core.so \
		${MKLROOT}/libmkl_intel_lp64.so \
		${MKLROOT}/libmkl_sequential.so \
		-lpthread -lm -ldl \
	     -lblas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -L/opt/intel/lib/intel64 -L/opt/intel/mkl/lib/intel64 

all: $(PY_OBJ)

$(PY_OBJ): matrix.o
	$(CXX) $(CXXFLAGS) $^  $(HEADERS) $(PIBINDFLAG) ${LDFLAGS}  -o $@

matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS)  $< $(HEADERS) $(PIBINDFLAG) ${LDFLAGS} -c -o $@

test: all
	python3 -m pytest -v -s

clean:
	rm *.o *.so __pycache__ .pytest_cache performance.txt -r -f