all: mp0 mp1 mp2-1 mp2-2

%: %.cu
	nvcc -o $@ -Ilibwb -L. -lwb $<

clean:
	-find . -maxdepth 1 -regex ".*/mp[0-9]+" -executable -print -delete
