all:
	nvcc -std=c++17 -O2 -o image_processor src/main.cu src/image_blur.cu -I./include

clean:
	rm -f image_processor
