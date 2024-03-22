# from /home/pie-perf/binaries_for_gem5.py
# %%
import subprocess 
import glob
import os 
import re
import sys
import warnings
VERBOSE = False

def add_abs_path_to_cpp_string(cpp_string, abs_path_to_project): 
    """
    line will have std::ifstream cin("some/path/to/input.txt"); 
    we prepend the absolute path to the input file
    """
    lines = cpp_string.split("\n")
    for i, line in enumerate(lines):
        if "std::ifstream cin" in line:
            lines[i] = re.sub('\(\"', f'(\"{abs_path_to_project}/', line)
    return "\n".join(lines)


def add_abs_path_to_cpp(cpp_file, abs_path_to_project): 
    """
    line will have std::ifstream cin("some/path/to/input.txt"); 
    we prepend the absolute path to the input file
    """
    with open(cpp_file, "r") as f:
        cpp_string = f.read()
    out_path = re.sub(".cpp", "_abs.cpp", cpp_file)
    with open(out_path, "w") as f:
        f.write(add_abs_path_to_cpp_string(cpp_string, abs_path_to_project))
    return out_path

def main(): 
    if len(sys.argv) > 1:
        ABS_PATH = sys.argv[1]
    else: 
        ABS_PATH = os.getcwd()
        warnings.warn(f"will treat {ABS_PATH} as the root of the project, otherwise pass the path as an argument")
    
    
    
    os.chdir(ABS_PATH)

    
    SAMPLES_PATH="data/samples_for_gem5/"
    INPUTS_PATH=SAMPLES_PATH + "input/"
    REFERENCE_PATH=SAMPLES_PATH + "reference/"
    GEN_PATH=SAMPLES_PATH + "gen/"

    inputs_cpp = glob.glob(INPUTS_PATH + "*.cpp")
    reference_cpp = glob.glob(REFERENCE_PATH + "*.cpp")
    generated_cpp = glob.glob(GEN_PATH + "*.cpp")

    print(f"there are {len(inputs_cpp)} inputs and {len(reference_cpp)} reference and {len(generated_cpp)} generated cpp programs")

        
    inputs_cpp_abs = []
    reference_cpp_abs = []
    generated_cpp_abs = []

    inputs_bin = []
    reference_bin = []
    generated_bin = []

    from tqdm import tqdm

    for type in ["inputs", "reference", "generated"]:
        paths = locals()[type + "_cpp"]
        paths = [path for path in paths if "_abs.cpp" not in path]
        for path in tqdm(paths, desc=f"redirecting and compiling {type}"):
            cpp_with_abs_path = add_abs_path_to_cpp(path, ABS_PATH)
            locals()[type + "_cpp_abs"].append(cpp_with_abs_path)
            p = subprocess.run(["g++", "--std=c++17",  "-O3", cpp_with_abs_path, "-o", re.sub(".cpp", ".out", cpp_with_abs_path)], capture_output=True)
            if VERBOSE:
                print(f"compiled {cpp_with_abs_path} to {re.sub('.cpp', '.out', cpp_with_abs_path)}")
            if p.returncode != 0:
                print(f"error compiling {cpp_with_abs_path} to {re.sub('.cpp', '.out', cpp_with_abs_path)}")
                print(p.stdout)
                print(p.stderr)
            locals()[type + "_bin"].append(re.sub(".cpp", ".out", cpp_with_abs_path))
            
    print(f"there are {len(inputs_bin)} inputs and {len(reference_bin)} reference and {len(generated_bin)} generated cpp programs")

        
        
        
            
    from tqdm import tqdm
    for bin_arr, bin_type in zip([inputs_bin, reference_bin, generated_bin], ["inputs", "reference", "generated"]):
        pbar = tqdm(total=len(bin_arr), desc=f"running {bin_type}")
        for bin in bin_arr:
            try: 
                p = subprocess.run([bin], capture_output=True, timeout=3)
                if p.returncode != 0:
                    print(f"error running {bin}")
                    print(p.stdout)
                    print(p.stderr)
            except subprocess.TimeoutExpired:
                print(f"timeout running {bin}")
                os.remove(bin)
                bin_arr.remove(bin)
                print(f"removed {bin}")
            pbar.update(1)
            


    print(f"there are {len(inputs_bin)} inputs and {len(reference_bin)} reference and {len(generated_bin)} generated cpp programs")


if __name__ == "__main__":
    main()