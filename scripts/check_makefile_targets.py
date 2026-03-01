import os
import re
import sys

def check_makefile(makefile_path="Makefile"):
    if not os.path.exists(makefile_path):
        print(f"Error: {makefile_path} not found.")
        return False

    with open(makefile_path, "r") as f:
        content = f.read()

    # Find all targets and their commands
    # This regex matches a target line followed by its command lines (starting with a tab)
    target_pattern = re.compile(r"^([a-zA-Z0-9_-]+):.*?## (.*?)\n((?:\t.*?(?:\n|$))*)", re.MULTILINE)
    targets = target_pattern.findall(content)
    
    all_passed = True
    for target, description, commands in targets:
        print(f"Checking target: {target} - {description}")
        for line in commands.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Check for python -m module
            match_m = re.search(r"\$\(PYTHON\) -m ([a-zA-Z0-9_.]+)", line)
            if not match_m:
                match_m = re.search(r"python -m ([a-zA-Z0-9_.]+)", line)
                
            if match_m:
                module = match_m.group(1)
                if module == "pytest":
                    print("  PASSED: Tool pytest assumed present.")
                    continue
                module_path = module.replace(".", "/") + ".py"
                if not os.path.exists(module_path):
                    # Check if it is a directory with __init__.py
                    if not (os.path.isdir(module.replace(".", "/")) and os.path.exists(module.replace(".", "/") + "/__init__.py")):
                        print(f"  FAILED: Module {module} (path: {module_path}) not found.")
                        all_passed = False
                    else:
                        print(f"  PASSED: Module {module} found.")
                else:
                    print(f"  PASSED: Module {module} found.")
                continue

            # Check for python scripts/some_script.py
            match_s = re.search(r"\$\(PYTHON\) (scripts/[a-zA-Z0-9_.]+.py)", line)
            if not match_s:
                match_s = re.search(r"python (scripts/[a-zA-Z0-9_.]+.py)", line)
                
            if match_s:
                script = match_s.group(1)
                if not os.path.exists(script):
                    print(f"  FAILED: Script {script} not found.")
                    all_passed = False
                else:
                    print(f"  PASSED: Script {script} found.")
                continue
    
    return all_passed

if __name__ == "__main__":
    if not check_makefile():
        sys.exit(1)
    sys.exit(0)
