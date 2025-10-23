#!/bin/bash
make
mkdir -p output/processed_images
./image_processor data/input_images output/processed_images | tee output/logs/execution.log
