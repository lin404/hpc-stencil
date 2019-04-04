#
# Makefile to build example MPI programs 
#

CC=mpiicc
COMP=GNU

ifeq ($(COMP), GNU)
  CFLAGS=-Wall -Ofast -no-multibyte-chars
endif

EXES=stencil


stencil: stencil.c
	$(CC) $(CFLAGS) -o $@ $^

all: $(EXES)

.PHONY: clean all

clean:
	\rm -f $(EXES) 
	\rm -f *.o
