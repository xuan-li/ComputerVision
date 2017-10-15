CC = g++
CFLAGS = -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib -l:libopencv_core.so.3.3.0 -l:libopencv_highgui.so.3.3.0 -l:libopencv_flann.so.3.3.0 -l:libopencv_imgproc.so.3.3.0 -l:libopencv_imgcodecs.so.3.3.0
SOURCEDIR = $(shell pwd)/Source
OUTPUTDIR = $(shell pwd)/bin
REQUIRMENTS = header.h main.cpp
TARGET = hw1
hw1: $(SOURCEDIR)/*
	mkdir $(OUTPUTDIR)
	$(CC) $(SOURCEDIR)/* -o $(OUTPUTDIR)/$(TARGET) $(CFLAGS)

.PHONY: clean

clean:
	rm -r $(OUTPUTDIR)