
CC = gcc
CFLAGS = -Wall -O3 -I../common_include -pthread 
TARGET = giantvm_slave
SRCS = slave_hybrid.c slave_vfio.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

---

## Step 7: 控制面工具 (Control Tool)

