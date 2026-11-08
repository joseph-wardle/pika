# Builds the bake tools against Houdini 22 into target/bake/: generate,
# which links its MaterialX, and texture, which links its OpenImageIO. The
# renderer keys each cache on its own tool, so rebuilding one never
# invalidates the other's. Houdini 21 has no Slang generator; a delegate
# built for Houdini 21 still runs these tools.

HFS ?= /opt/hfs22.0
ROOT := $(abspath ..)
OUT := $(ROOT)/target/bake

CXXFLAGS := -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Wall -Wextra \
	-isystem $(HFS)/toolkit/include -DHFS='"$(HFS)"' -DBAKE='"$(abspath .)"'
# A DT_RPATH rather than a DT_RUNPATH: it beats the LD_LIBRARY_PATH a
# Houdini 21 host sets, which holds an older MaterialX of the same soname.
LDFLAGS := -pthread -L$(HFS)/dsolib -Wl,-rpath,$(HFS)/dsolib -Wl,--disable-new-dtags

all: $(OUT)/generate $(OUT)/texture

$(OUT)/generate: $(OUT)/generate.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lMaterialXGenSlang -lMaterialXGenHw -lMaterialXGenShader \
		-lMaterialXFormat -lMaterialXCore

$(OUT)/texture: $(OUT)/texture.o $(OUT)/bc7enc.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lOpenImageIO_sidefx -lOpenImageIO_Util_sidefx

$(OUT)/%.o: %.cpp
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Third-party code (bc7enc/NOTICE), built without Pika's warnings.
$(OUT)/bc7enc.o: bc7enc/bc7enc.cpp bc7enc/bc7enc.h
	mkdir -p $(OUT)
	$(CXX) -std=c++17 -O2 -c -o $@ $<

.PHONY: all
