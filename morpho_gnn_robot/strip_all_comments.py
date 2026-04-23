import ast
import os
import glob

def remove_docstrings(node):
    for child in ast.walk(node):
        if not isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        if not len(child.body):
            continue
        if not isinstance(child.body[0], ast.Expr):
            continue
        if not hasattr(child.body[0], 'value') or not isinstance(child.body[0].value, ast.Constant):
            continue
        if isinstance(child.body[0].value.value, str):
            child.body = child.body[1:]
    return node

def strip_comments_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        parsed = ast.parse(source)
    except SyntaxError as e:
        print(f"SyntaxError in {filename}: {e}")
        return
        
    parsed = remove_docstrings(parsed)
    new_source = ast.unparse(parsed)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_source)

if __name__ == "__main__":
    count = 0
    for root, dirs, files in os.walk("."):
        if "build" in dirs:
            dirs.remove("build")
        if "install" in dirs:
            dirs.remove("install")
        if "log" in dirs:
            dirs.remove("log")
        if "test" in dirs:
            dirs.remove("test") # Also exclude standard ros test files like test_copyright.py
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if file != "strip_all_comments.py" and file != "test_strip.py":
                    print(f"Stripping {path}...")
                    strip_comments_from_file(path)
                    count += 1
    print(f"Stripped comments from {count} files.")
