stencil: stencil.c
	icc -std=c99 -Wall $^ -o $@ -no-multibyte-chars -Ofast
