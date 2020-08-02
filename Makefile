CPP = g++
CPPFLAGS = -O2 -g -Wall -Wshadow -fPIC -fopenmp -pipe -march=native -mfpmath=sse -msse2 -ffast-math -std=c++11# -v
CPPDEBUG = -O0 -g
CPPGPROF = -O0 -pg

NVFLAGS  = -O2 -g -gencode arch=compute_61,code=sm_61 --use_fast_math -std=c++11

INC = -I /usr/local/cuda/include -I /opt/intel/mkl/include
LIB = -Wl,-dy -lmysqlclient -lcityhash -lconfig++ -lglog -lgflags -llz4 \
      -Wl,-dy -L /opt/intel/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
      -Wl,-dy -L /opt/intel/lib/intel64_lin/ -liomp5 \
      -Wl,-dy -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcusparse -lcudnn -lnccl \
      -Wl,-dy -lz -ldl -lm -lstdc++fs
LIBXXX = -Wl,-dn -lmysqlclient -lcityhash -lconfig++ -lglog -lgflags -llz4 \
      -Wl,--start-group \
      /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a \
      /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a \
      /opt/intel/mkl/lib/intel64/libmkl_core.a \
      /opt/intel/lib/intel64_lin/libiomp5.a \
      -Wl,--end-group \
      -Wl,-dy -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcusparse -lcudnn -lnccl \
      -Wl,-dy -lssl -lcrypto -lz -ldl -lm -lstdc++fs

CV_INC = `pkg-config opencv --cflags` $(INC)
CV_LIB = `pkg-config opencv --libs`   $(LIB)

UT_DIR = ./base/
DF_DIR = ./dataformat/
IM_DIR = ./image/ # ./segment/
TS_DIR = ./tensor/
ML_DIR = ./optimization/
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

PROGRAMS = df_imagenet df_coco_seg nnetMain nnetInfer 
all: $(PROGRAMS)

clean:
	rm df_imagenet df_coco_seg nnetMain nnetInfer *.a
cleanobj:
	@rm $(UT_OBJ) $(IM_OBJ) $(TS_OBJ) $(TS_CUO) $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)



$(UT_OBJ) : %.o : %.cpp ./include/base.h
	$(CPP) $(CPPFLAGS) $(INC) -c $< -o $@

$(TS_OBJ)  : %.o   : %.cpp ./include/tensor.h ./include/xpu.h ./include/expr.h
	$(CPP) $(CPPFLAGS) $(INC) -c $< -o $@
$(ML_OBJ)  : %.o   : %.cpp ./include/optimization.h
	$(CPP) $(CPPFLAGS) $(INC) -c $< -o $@
$(NN_OBJ)  : %.o   : %.cpp ./include/nnet.h
	$(CPP) $(CPPFLAGS) $(INC) -c $< -o $@

$(TS_CUO)  : %.cuo : %.cpp ./include/tensor.h ./include/xpu.h ./include/expr.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@
$(ML_CUO)  : %.cuo : %.cpp ./include/optimization.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@
$(NN_CUO)  : %.cuo : %.cpp ./include/nnet.h
	nvcc -x cu -ccbin=g++ -Xcompiler -fPIC -DNDEBUG $(NVFLAGS) $(INC) -c $< -o $@


$(IM_OBJ)  : %.o : %.cpp ./include/image.h
	$(CPP) $(CPPFLAGS) $(CV_INC) -c $< -o $@

libtensor.so: $(TS_OBJ) $(TS_CUO)
	@$(CPP) -shared -fPIC -o $@ $(TS_OBJ) $(TS_CUO)
libnnet.so:   $(NN_OBJ) $(NN_CUO)
	@$(CPP) -shared -fPIC -o $@ $(NN_OBJ) $(NN_CUO)

libtensor.a: $(TS_OBJ) $(TS_CUO)
	@ar -cr $@ $(TS_OBJ) $(TS_CUO)
libml.a:     $(ML_OBJ) $(ML_CUO)
	@ar -cr $@ $(ML_OBJ) $(ML_CUO)
libnnet.a:   $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)
	@ar -cr $@ $(ML_OBJ) $(ML_CUO) $(NN_OBJ) $(NN_CUO)



df_imagenet: ./dataformat/df_imagenet.cpp $(UT_OBJ) $(IM_OBJ) libtensor.a
	@$(CPP) $(CPPFLAGS) $(CV_INC) ./dataformat/$@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -ltensor $(CV_LIB)

df_coco_seg: ./dataformat/df_coco_seg.cpp $(UT_OBJ) $(IM_OBJ) libtensor.a
	@$(CPP) $(CPPFLAGS) $(CV_INC) ./dataformat/$@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -ltensor $(CV_LIB)



knnBrute:  knnBrute.cpp  $(UT_OBJ) libtensor.a libml.a
	$(CPP) $(CPPFLAGS) $(CV_INC) $@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -lml -L. -ltensor $(CV_LIB)

knnGraph:  knnGraph.cpp  $(UT_OBJ) libtensor.a libml.a
	$(CPP) $(CPPFLAGS) $(CV_INC) $@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -lml -L. -ltensor $(CV_LIB)

nnetMain:  nnetMain.cpp  $(UT_OBJ) $(IM_OBJ) libtensor.a libnnet.a
	$(CPP) $(CPPFLAGS) $(CV_INC) $@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -lnnet -L. -ltensor $(CV_LIB)

nnetInfer:  nnetInfer.cpp  $(UT_OBJ) $(IM_OBJ) libtensor.a libnnet.a
	$(CPP) $(CPPFLAGS) $(CV_INC) $@.cpp -o $@ $(UT_OBJ) $(IM_OBJ) -L. -lnnet -L. -ltensor $(CV_LIB)
