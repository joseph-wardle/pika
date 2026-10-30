# Builds the bake tool against Houdini 22's MaterialX into target/bake/.
# Houdini 21 has no Slang generator; a delegate built for Houdini 21 still
# runs this tool.

HFS ?= /opt/hfs22.0
ROOT := $(abspath ..)
OUT := $(ROOT)/target/bake

CXXFLAGS := -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Wall -Wextra \
	-isystem $(HFS)/toolkit/include -DHFS='"$(HFS)"' -DBAKE='"$(abspath .)"'
# A DT_RPATH rather than a DT_RUNPATH: it beats the LD_LIBRARY_PATH a
# Houdini 21 host sets, which holds an older MaterialX of the same soname.
LDFLAGS := -L$(HFS)/dsolib -Wl,-rpath,$(HFS)/dsolib -Wl,--disable-new-dtags
LIBS := -lMaterialXGenSlang -lMaterialXGenHw -lMaterialXGenShader -lMaterialXFormat \
	-lMaterialXCore

$(OUT)/bake: bake.cpp
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)
