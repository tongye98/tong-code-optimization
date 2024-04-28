def is_answer_correct(input_case_path, stdout_bin):
    """
    check stdout_bin is correct 
    """
    output_case_path = input_case_path.replace("input", "output")
    with open(output_case_path, 'r') as g:
        truth = g.read()
    
    ground_truth_lines = truth.strip().splitlines()
    print(f"ground_truth = {ground_truth_lines}")
    output_lines = stdout_bin.strip().splitlines()
    print(f"output = {output_lines}")

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

if __name__ == "__main__":
    input_case_path = "testest.txt"
    stdout_bin = "\ndalfdla"

    print(is_answer_correct(input_case_path, stdout_bin))
