import nbformat
from nbformat import v4 as nbf
import re
from pathlib import Path

# Ensure output folder exists
output_dir = Path("notebooks")
output_dir.mkdir(exist_ok=True)

# Find all markdown files in repo
md_files = list(Path(".").rglob("*.md"))

for md_path in md_files:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split between markdown and code
    parts = re.split(r"```(?:python)?", content)
    cells = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            cells.append(nbf.new_markdown_cell(part.strip()))
        else:
            code = part.split("```")[0]  # remove closing ```
            cells.append(nbf.new_code_cell(code.strip()))

    # Create notebook
    nb = nbf.new_notebook(cells=cells, metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    })

    # Use filename stem for .ipynb name ->   output_file = output_dir / (md_path.stem + ".ipynb")
    #  - New notebook name is prepended with "nb-" and then the 
    #   original stem forced to all lowercase

    output_file = output_dir / f"nb-{md_path.stem.lower()}.ipynb"
    with open(output_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Converted: {md_path} → {output_file}")


'''
import nbformat
from nbformat import v4 as nbf
import re

# Load README.md content
with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# Split between markdown and code
parts = re.split(r"```(?:python)?", content)
cells = []
for i, part in enumerate(parts):
    if i % 2 == 0:
        cells.append(nbf.new_markdown_cell(part.strip()))
    else:
        code = part.split("```")[0]  # remove closing ```
        cells.append(nbf.new_code_cell(code.strip()))

# Create notebook
nb = nbf.new_notebook(cells=cells, metadata={
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
})

# Save to file
with open("monte_carlo_tutorial.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook saved as monte_carlo_tutorial.ipynb")
'''