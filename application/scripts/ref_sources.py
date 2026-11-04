"""Print the reference's source files, as listed in prepare.py (SRC_FILES)."""
import ast, re, sys
text = open(sys.argv[1]).read()
files = ast.literal_eval(re.search(r"SRC_FILES\s*=\s*(\[.*?\]|\(.*?\))", text, re.S).group(1))
print(" ".join("src/" + f for f in files if not f.endswith((".h", ".hpp"))))
