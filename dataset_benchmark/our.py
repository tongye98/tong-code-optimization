
import subprocess
import shlex

cmd = f"/home/gem5/build/X86/gem5.opt --stats-file='temp.txt'  /home/gem5-skylake-config/gem5-configs/run-se.py Verbatim home/working_dir/our.out"
print(f'GEM5 executing {cmd}')
cmd_args = shlex.split(cmd)
with open('input.txt', 'r') as fh:
    p = subprocess.run(cmd_args,
                        # preexec_fn=limit_virtual_memory,
                        capture_output=True,
                        # bufsize=MAX_VIRTUAL_MEMORY,
                        timeout=120,
                        stdin=fh,
                        text=True
                        )
    returncode = p.returncode
    stdout = p.stdout
    stderr = p.stderr
    print(returncode)
    print(stdout)
    print(stderr)