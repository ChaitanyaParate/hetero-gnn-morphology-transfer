import ast

def strip_comments(filename):
    with open(filename, 'r') as f:
        source = f.read()
    
    parsed = ast.parse(source)
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        if not len(node.body):
            continue
        if not isinstance(node.body[0], ast.Expr):
            continue
        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
            continue
        # It's a docstring, remove it
        node.body = node.body[1:]
        
    return ast.unparse(parsed)

print(strip_comments('Training_Location/test_stand.py'))
