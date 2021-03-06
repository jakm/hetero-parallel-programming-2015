all: mp0 mp1 mp2-1 mp2-2 mp3 mp4 mp5 mp6 mp7 mp8

mp8: mp8.cpp
	nvcc -arch=sm_30 -lOpenCL -o $@ -Ilibwb -L. -lwb $<

%: %.cu
	nvcc -arch=sm_30 -o $@ -Ilibwb -L. -lwb $<

clean:
	-find . -maxdepth 1 -regex ".*/mp[0-9]+" -executable -print -delete
