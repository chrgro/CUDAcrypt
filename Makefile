COMPILER := nvcc

## Makefile for cudacrypt
CU_FILES := $(wildcard src/*.cu)
H_FILES := $(wildcard src/*.h)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))


default : bin/cudacrypt

bin/cudacrypt : $(OBJ_FILES) $(H_FILES) 
	$(COMPILER) -o $@ $(OBJ_FILES)
	
	
obj/%.o : src/%.cu $(H_FILES)
	$(COMPILER) -c -o $@ $< -I/src


## Makefile for testing
## (More naive compiling than above)

TEST_CU_FILES := $(wildcard test/*.cu)
TEST_H_FILES := $(wildcard test/*.h)
#OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

test : bin/test
bin/test : $(TEST_CU_FILES) $(H_FILES) $(OBJ_FILES)
	$(COMPILER) -o $@ $(OBJ_FILES) $(TEST_CU_FILES)
	