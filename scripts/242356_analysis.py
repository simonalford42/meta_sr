import ast
from pathlib import Path

path = Path(__file__).parent.parent / "out" / "242356_scratch.txt"

lists = []
for line in path.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    bracket = line[line.index("["):]
    lists.append(ast.literal_eval(bracket))

mapping = {i: [idx for idx, lst in enumerate(lists) if i in lst] for i in range(21)}

for i in range(21):
    print(f"{i}: {mapping[i]}")
