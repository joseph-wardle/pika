# Builds the bake tools against Houdini 22 into target/bake/. Houdini 21 has no
# Slang generator; a delegate built for Houdini 21 still runs these tools.

HFS ?= /opt/hfs22.0
ISPC ?= ispc
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

$(OUT)/texture: $(OUT)/texture.o $(OUT)/bc7e.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lOpenImageIO_sidefx -lOpenImageIO_Util_sidefx

$(OUT)/texture.o: $(OUT)/bc7e.o

$(OUT)/%.o: %.cpp
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -I$(OUT) -c -o $@ $<

# Third-party code (bc7e/NOTICE). One ispc target rather than its dispatch, so
# every machine bakes the same bytes.
$(OUT)/bc7e.o: bc7e/bc7e.ispc
	mkdir -p $(OUT)
	$(ISPC) -O2 --pic --target=avx2-i32x8 --opt=disable-assertions --woff -o $@ -h $(OUT)/bc7e_ispc.h $<

.PHONY: all
