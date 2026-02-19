import json
import glob
import re

def sanitize_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "sk-proj-" in line:
                    # Replace with placeholder
                    # Keep indentation if possible
                    indent = line[:line.find("sk-proj-")] if "sk-proj-" in line else ""
                    # Simple replacement logic: replace the whole string literal
                    # assuming it's like api_key = "sk-..."
                    line = re.sub(r'["\']sk-proj-[a-zA-Z0-9_\-]+["\']', 'os.getenv("OPENAI_API_KEY")', line)
                    modified = True
                new_source.append(line)
            cell['source'] = new_source
            
    if modified:
        print(f"Sanitized {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
    else:
        print(f"No keys found in {filepath}")

notebooks = glob.glob("*.ipynb")
for nb in notebooks:
    sanitize_notebook(nb)
