
import subprocess
import shlex
import json

result = [
    {'a': {"aa": 1, "bb": 2, "cc": 3}},
    {'b': {"aa": 1, "bb": 2, "cc": 3}},
    {'c': {"aa": 1, "bb": 2, "cc": 3}}
]
print(result)

with open('result.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)