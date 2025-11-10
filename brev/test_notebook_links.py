import os
import sys
import json
import argparse
import glob
from pathlib import Path
from link_validator import find_md_links, validate_links

def extract_markdown_from_notebook(file_path):
    """
    Reads a Jupyter notebook and extracts all markdown content from markdown cells.
    Returns the combined markdown content as a single string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading notebook file {file_path}: {e}")
        return None

    markdown_content = []
    
    # Extract markdown cells
    if 'cells' in notebook:
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'markdown':
                # Join the source lines (which can be a list or a string)
                source = cell.get('source', [])
                if isinstance(source, list):
                    markdown_content.append(''.join(source))
                else:
                    markdown_content.append(source)
    
    return '\n\n'.join(markdown_content)

def validate_notebook_links(file_path):
    """
    Reads a Jupyter notebook, extracts all links from markdown cells, and validates them.
    """
    markdown_content = extract_markdown_from_notebook(file_path)
    
    if markdown_content is None:
        return False
    
    if not markdown_content.strip():
        print(f"No markdown content found in {file_path}.")
        return True

    links = find_md_links(markdown_content)
    base_dir = os.path.dirname(os.path.abspath(file_path))

    return validate_links(links, base_dir, file_path)

def collect_notebook_files(path_pattern):
    """
    Collects notebook files based on the provided path pattern.
    Supports:
    - Single file path
    - Directory path (finds all .ipynb files recursively)
    - Glob patterns (e.g., *.ipynb, notebooks/**/*.ipynb)
    """
    files = []
    path = Path(path_pattern)
    
    # Check if it's an exact file match
    if path.is_file():
        files.append(str(path))
    # Check if it's a directory
    elif path.is_dir():
        # Find all .ipynb files recursively
        files.extend([str(f) for f in path.rglob('*.ipynb')])
    else:
        # Try glob pattern matching
        matched_files = glob.glob(path_pattern, recursive=True)
        # Filter to only notebook files
        files.extend([f for f in matched_files if f.endswith('.ipynb') and os.path.isfile(f)])
    
    return sorted(files)

# Example Usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate all links in Jupyter notebook(s).",
        epilog="Examples:\n"
               "  %(prog)s notebook.ipynb               # Single file\n"
               "  %(prog)s notebooks/                   # All .ipynb files in directory\n"
               "  %(prog)s '*.ipynb'                    # All .ipynb files in current dir\n"
               "  %(prog)s 'notebooks/**/*.ipynb'       # All .ipynb files recursively\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path to notebook file, directory, or glob pattern")
    args = parser.parse_args()
    
    # Collect all files to process
    files_to_check = collect_notebook_files(args.path)
    
    if not files_to_check:
        print(f"No notebook files found matching: {args.path}")
        sys.exit(1)
    
    print(f"Found {len(files_to_check)} notebook file(s) to check.\n")
    
    # Process all files and track results
    results = {}
    for file_path in files_to_check:
        print(f"\n{'='*80}")
        print(f"Checking: {file_path}")
        print('='*80)
        results[file_path] = validate_notebook_links(file_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    passed = sum(1 for result in results.values() if result)
    failed = len(results) - passed
    
    for file_path, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {file_path}")
    
    print(f"\nTotal: {len(results)} files | Passed: {passed} | Failed: {failed}")
    
    if failed > 0:
        print("\nTest failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)

