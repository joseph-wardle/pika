# Builds the Pika Hydra delegate into target/hydra/

ROOT := $(abspath ..)
OUT := $(ROOT)/target/hydra
RUST := $(ROOT)/target/release/libpika.a

CXXFLAGS := -std=c++17 -fPIC -O2 -Wall -Wextra -Wno-deprecated
# --exclude-libs keeps Rust's symbols, std included
LDFLAGS := -shared -Wl,--exclude-libs,ALL -Wl,--no-undefined

ifdef HFS
# Houdini 21 ships Python 3.11 and Houdini 22 ships 3.13.
PYTHON := $(notdir $(wildcard $(HFS)/toolkit/include/python3.*))
# The HDK's ABI flags: Houdini's USD is built with the C++11 string ABI.
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1 \
	-isystem $(HFS)/toolkit/include -isystem $(HFS)/toolkit/include/$(PYTHON)
USD := -L$(HFS)/dsolib -lpxr_hd -lpxr_hdsi -lpxr_hf -lpxr_cameraUtil -lpxr_pxOsd -lpxr_sdf \
	-lpxr_vt -lpxr_gf -lpxr_tf -lpxr_trace -lpxr_arch -lpxr_python \
	-L$(HFS)/python/lib -l$(PYTHON)
else
CXXFLAGS += $(shell python3-config --includes)
USD := -lusd_ms $(shell python3-config --ldflags --embed)
endif
LIBS := $(USD) -losdCPU -ldl -lpthread -lm -lrt -lutil -lgcc_s

SOURCES := plugin.cpp prims.cpp

$(OUT)/pika.so: $(SOURCES) $(wildcard *.h) $(RUST) $(OUT)/plugInfo.json $(OUT)/hfs
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(RUST) $(LIBS)

# The Houdini the delegate was last built against
$(OUT)/hfs: FORCE
	@mkdir -p $(OUT)
	@echo $(HFS) | cmp -s - $@ || echo $(HFS) > $@

$(OUT)/plugInfo.json: plugInfo.json
	mkdir -p $(OUT)
	cp $< $@

# Cargo decides whether the library is stale.
$(RUST): FORCE
	cd $(ROOT) && cargo build --locked --release --lib

FORCE:
.PHONY: FORCE
