CC = gcc
CFLAGS = -Wall -O3 -msse4.2 -I../common_include -pthread 
TARGET = wavevm_node_slave
SRCS = slave_hybrid.c slave_vfio.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
