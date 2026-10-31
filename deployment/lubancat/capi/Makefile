CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra
INCLUDES := -Iinclude
LDLIBS := -lrknnrt -lm

.PHONY: all clean

all: libp2capi.so

libp2capi.so: p2capi.c include/rknn_api.h
	$(CC) $(CFLAGS) $(INCLUDES) -shared p2capi.c $(LDLIBS) -o $@

clean:
	rm -f libp2capi.so

# RK3566 board build used by the live AI bridge:
#   make CC=gcc
# The resulting libp2capi.so is loaded by capi_detector.py through ctypes.
# Link against the board's librknnrt (RKNN Runtime 2.3.2); do not bundle a
# different runtime with the library.
