all: 5mer/libhmm.so 7mer/libhmm.so

5mer/libhmm.so: 5mer/hmm.cu
	nvcc $< -O3 -arch=native -Xcompiler -fPIC --shared -o $@

7mer/libhmm.so: 7mer/hmm.cu
	nvcc $< -O3 -arch=native -Xcompiler -fPIC --shared -o $@

clean:
	rm -rf *.stdout *.stderr *.pdf 5mer/libhmm.so 7mer/libhmm.so __pycache__

clean-all: clean
	rm -f *results*.csv
