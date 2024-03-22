#! /bin/bash

python test_hello.py \
--file_path "/home/tongye/code_generation/tong-code-optimization/test_gem5/hello_world.cpp" \
--executable_path "/home/tongye/code_generation/tong-code-optimization/test_gem5/hello_world.out" \
--gem5_path "/home/tongye/code_generation/gem5/build/X86/gem5.opt" \
--gem5_script "/home/tongye/code_generation/gem5-skylake-config/gem5-configs/run-se.py" \