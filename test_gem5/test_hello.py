import argparse
import subprocess

def compile_hello_world(file_path, output_path, timeout=None):
    compile_command = ['g++', '-std=c++17', '-O3', file_path, '-o', output_path]
    subprocess.run(compile_command, check=True, timeout=timeout)

def run_hello_world(executable_path, timeout=None):
    run_command = [executable_path]
    subprocess.run(run_command, check=True, timeout=timeout)

def run_hello_world_with_gem5(executable_path, gem5_path, gem5_script, timeout=None):
    run_command = [gem5_path, gem5_script, 'Verbatim', executable_path]
    subprocess.run(run_command, check=True, timeout=timeout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hello World Python Script')
    parser.add_argument('--file_path', type=str, help='Path to hello_world.cpp')
    parser.add_argument('--executable_path', type=str, help='Path to compiled executable')
    parser.add_argument('--gem5_path', type=str, help='Path to gem5.opt')
    parser.add_argument('--gem5_script', type=str, help='Path to gem5 script')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout for each subprocess')

    args = parser.parse_args()

    # Compiling hello_world.cpp
    print('Compiling hello_world.cpp with g++')
    compile_hello_world(args.file_path, args.executable_path, args.timeout)

    # Running hello_world.out
    print('Running hello_world.out')
    run_hello_world(args.executable_path, args.timeout)

    # Running hello_world.out with gem5
    print('Running hello_world.out with gem5')
    run_hello_world_with_gem5(args.executable_path, args.gem5_path, args.gem5_script, args.timeout)
