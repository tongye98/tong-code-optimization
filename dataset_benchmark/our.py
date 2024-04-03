
import subprocess
import shlex

def get_accuracy(stdout, truth):
    """
    Compare the output of the code with the ground truth.
    """
    ground_truth_lines = truth.strip().splitlines()
    output_lines = stdout.strip().splitlines()
    print(ground_truth_lines)
    print(output_lines)
    IsCorrect = True
    for gen_output, ground_truth_output in zip(output_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        
        if not is_corr:
            IsCorrect = False
    
    return IsCorrect

stdout_bin = "1 2 3 3 3 3 4 2 2 3  "
ground_truth = "1\n2\n3\n3\n3\n3\n4\n2\n2\n3"


print(get_accuracy(stdout_bin, ground_truth))