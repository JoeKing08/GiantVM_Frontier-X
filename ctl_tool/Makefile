
CC = gcc
CFLAGS = -Wall -O2 -I../common_include
TARGET = gvm_ctl

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

