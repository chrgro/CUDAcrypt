
COMPILER := nvcc
CU_FILES := $(wildcard src/*.cu)
H_FILES := $(wildcard src/*.h)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

default : bin/cudacrypt

bin/cudacrypt : $(OBJ_FILES) $(H_FILES)
	$(COMPILER) -o $@ $(OBJ_FILES)
	
	
obj/%.o : src/%.cu $(H_FILES)
	$(COMPILER) -c -o $@ $< -I/src


