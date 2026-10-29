# Builds the Pika Hydra delegate against Houdini's USD (the HDK) into
# target/hydra/, ready for PXR_PLUGINPATH_NAME.

HFS ?= /opt/hfs21.0
ROOT := $(abspath ..)
OUT := $(ROOT)/target/hydra
RUST := $(ROOT)/target/release/libpika.a

# The HDK's ABI flags: Houdini's USD is built with the C++11 string ABI.
CXXFLAGS := -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC -O2 -Wall -Wextra \
	-Wno-deprecated \
	-isystem $(HFS)/toolkit/include -isystem $(HFS)/toolkit/include/python3.11
# --exclude-libs keeps Rust's symbols, std included, private to pika.so so
# they cannot clash with another library loaded into the host.
LDFLAGS := -shared -Wl,--exclude-libs,ALL -Wl,--no-undefined
# Houdini's pxr.h turns Python support on, so VtValue pulls in Python.
LIBS := -L$(HFS)/dsolib -lpxr_hd -lpxr_hf -lpxr_cameraUtil -lpxr_pxOsd -losdCPU -lpxr_sdf -lpxr_vt -lpxr_gf \
	-lpxr_tf -lpxr_trace -lpxr_arch -lpxr_python \
	-L$(HFS)/python/lib -lpython3.11 -ldl -lpthread -lm -lrt -lutil -lgcc_s

SOURCES := plugin.cpp prims.cpp

$(OUT)/pika.so: $(SOURCES) $(wildcard *.h) $(RUST) $(OUT)/plugInfo.json
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(RUST) $(LIBS)

$(OUT)/plugInfo.json: plugInfo.json
	mkdir -p $(OUT)
	cp $< $@

# Cargo decides whether the library is stale.
$(RUST): FORCE
	cd $(ROOT) && cargo build --locked --release --lib

FORCE:
.PHONY: FORCE
