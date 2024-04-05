
import subprocess
import shlex
import json

test = {
        "s268517277": {
            "problem_id": "p03275",
            "user_id": "u863370423",
            "submission_id": "s268517277",
            "test_case_id": "32",
            "sim_seconds": 0.00019,
            "sim_ticks": 189955194,
            "sim_seconds_precise": 0.000189955194
        }
    }
print(type(test))
print(test.keys())
print(test.values())
print(list(test.values())[0]["sim_seconds_precise"])