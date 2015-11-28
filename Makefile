CXX = g++
CXXFLAGS = -O2 -g -Wall -Wshadow \
	   -fPIC -fopenmp -pipe -march=native -mfpmath=sse -msse2 -ftracer -fivopts -fforce-addr -std=c++11# -v
CXXDEBUG = -O0 -g
CXXGPROF = -O0 -pg

NVFLAGS  = -O2 -g -G -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_35,code=sm_35 -std=c++11

INC = -I /usr/local/cuda/include -I /opt/intel/mkl/include
LIB = -Wl,-dy -lconfig++ -lglog -lgflags \
      -Wl,-dy -L /opt/intel/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
      -Wl,-dy -L /opt/intel/lib/intel64/ -liomp5 \
      -Wl,-dy -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcusparse -lcudnn \
      -Wl,-dy -ldl -lm
LIBXXX = -Wl,-dn -lconfig++ -lglog -lgflags \
      -Wl,--start-group \
      /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
      /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a \
      /opt/intel/mkl/lib/intel64/libmkl_core.a \
      /opt/intel/lib/intel64/libiomp5.a \
      -Wl,--end-group \
      -Wl,-dy -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcusparse -lcudnn \
      -Wl,-dy -lssl -lcrypto -ldl -lm

CV_INC = `pkg-config opencv --cflags` $(INC)
CV_LIB = `pkg-config opencv --libs`   $(LIB)

UT_DIR = ./util/
DF_DIR = ./dataformat/
IM_DIR = ./image/
TS_DIR = ./tensor/
ML_DIR = ./learning/ ./optimization/
NN_DIR = ./nnet/

UT_SRC = $(foreach dir, $(UT_DIR), $(wildcard $(dir)*.cpp))
IM_SRC = $(foreach dir, $(IM_DIR), $(wildcard $(dir)*.cpp))
TS_SRC = $(foreach dir, $(TS_DIR), $(wildcard $(dir)*.cpp))
ML_SRC = $(foreach dir, $(ML_DIR), $(wildcard $(dir)*.cpp))
NN_SRC = $(foreach dir, $(NN_DIR), $(wildcard $(dir)*.cpp))

UT_OBJ = $(patsubst %.cpp, %.o, $(UT_SRC))
IM_OBJ = $(patsubst %.cpp, %.o, $(IM_SRC))
TS_OBJ = $(patsubst %.cpp, %.o, $(TS_SRC))
ML_OBJ = $(patsubst %.cpp, %.o, $(ML_SRC))
NN_OBJ = $(patsubst %.cpp, %.o, $(NN_SRC))
TS_CUO = $(patsubst %.cpp, %.cuo, $(TS_SRC))
ML_CUO = $(patsubst %.cpp, %.cuo, $(ML_SRC))
NN_CUO = $(patsubst %.cpp, %.cuo, $(NN_SRC))



.PHONY: clean cleanobj

PROGRAMS = df_image df_mean nnetMain

all: $(PROGRAMS)

clean:
	rm df_image df_mean nnetMain *.a
cleanobj:
	@rm $(UT_OBJ) $(IM_OBJ) $(TS_OBJ) $(TS_CUO) $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)



$(UT_OBJ) : %.o : %.cpp ./include/util.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(TS_OBJ)  : %.o   : %.cpp ./include/tensor.h ./include/xpu.h ./include/expr.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@
$(ML_OBJ)  : %.o   : %.cpp ./include/optimization.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@
$(NN_OBJ)  : %.o   : %.cpp ./include/nnet.h
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(TS_CUO)  : %.cuo : %.cpp ./include/tensor.h ./include/xpu.h ./include/expr.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@
$(ML_CUO)  : %.cuo : %.cpp ./include/optimization.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@
$(NN_CUO)  : %.cuo : %.cpp ./include/nnet.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@


$(IM_OBJ)  : %.o : %.cpp ./include/image.h
	$(CXX) $(CXXFLAGS) $(CV_INC) -c $< -o $@

libtensor.so: $(TS_OBJ) $(TS_CUO)
	@$(CXX) -shared -fPIC -o $@ $(TS_OBJ) $(TS_CUO)
libnnet.so:   $(NN_OBJ) $(NN_CUO)
	@$(CXX) -shared -fPIC -o $@ $(NN_OBJ) $(NN_CUO)

libtensor.a: $(TS_OBJ) $(TS_CUO)
	@ar -cr $@ $(TS_OBJ) $(TS_CUO)
libml.a:     $(ML_OBJ) $(ML_CUO)
	@ar -cr $@ $(ML_OBJ) $(ML_CUO)
libnnet.a:   $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)
	@ar -cr $@ $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)



df_mean:  ./dataformat/df_mean.cpp  $(UT_OBJ) $(IM_OBJ) libtensor.a
	@$(CXX) $(CXXFLAGS) $(CV_INC) ./dataformat/$@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -ltensor $(CV_LIB)

df_image: ./dataformat/df_image.cpp $(UT_OBJ) $(IM_OBJ) libtensor.a
	@$(CXX) $(CXXFLAGS) $(CV_INC) ./dataformat/$@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -ltensor $(CV_LIB)

nnetMain:  nnetMain.cpp  $(UT_OBJ) $(IM_OBJ) libtensor.a libnnet.a
	$(CXX) $(CXXFLAGS) $(CV_INC) $@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -lnnet -L. -ltensor $(CV_LIB)

