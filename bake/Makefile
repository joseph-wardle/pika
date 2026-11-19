# Builds the bake tools into target/bake/: against the MaterialX and OpenImageIO on the compiler's
# search path, or against Houdini 22's when HFS names it. Houdini 21 has no Slang generator.

ISPC ?= ispc
ROOT := $(abspath ..)
OUT := $(ROOT)/target/bake

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -DBAKE='"$(abspath .)"'
LDFLAGS := -pthread

ifdef HFS
MATERIALX := $(HFS)/houdini/materialx
OIIO := -lOpenImageIO_sidefx -lOpenImageIO_Util_sidefx
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1 -isystem $(HFS)/toolkit/include
LDFLAGS += -L$(HFS)/dsolib -Wl,-rpath,$(HFS)/dsolib -Wl,--disable-new-dtags
else
ifndef MATERIALX
$(error Set MATERIALX to the directory holding MaterialX's libraries/, or HFS to a Houdini 22)
endif
OIIO := -lOpenImageIO -lOpenImageIO_Util
endif
CXXFLAGS += -DMATERIALX='"$(MATERIALX)"'

all: $(OUT)/generate $(OUT)/texture

$(OUT)/generate: $(OUT)/generate.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lMaterialXGenSlang -lMaterialXGenHw -lMaterialXGenShader \
		-lMaterialXFormat -lMaterialXCore

$(OUT)/texture: $(OUT)/texture.o $(OUT)/bc7e.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(OIIO)

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
