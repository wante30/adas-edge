CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra
RKNN_INC ?= ../../../deployment/lubancat/capi/include
LDLIBS := -lrknnrt -lm

.PHONY: all clean

all: libp2capi_generic.so rknn_bench_generic

libp2capi_generic.so: p2capi_generic.c $(RKNN_INC)/rknn_api.h
	$(CC) $(CFLAGS) -I$(RKNN_INC) -shared p2capi_generic.c $(LDLIBS) -o $@

rknn_bench_generic: rknn_bench_generic.c $(RKNN_INC)/rknn_api.h
	$(CC) $(CFLAGS) -I$(RKNN_INC) rknn_bench_generic.c $(LDLIBS) -o $@

clean:
	rm -f libp2capi_generic.so rknn_bench_generic
