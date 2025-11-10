import os
import sys
import argparse
import glob
from pathlib import Path
from link_validator import find_md_links, validate_links

def validate_markdown_links(file_path):
    """
    Reads a Markdown file, extracts all links, and validates them.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return False

    links = find_md_links(content)
    base_dir = os.path.dirname(os.path.abspath(file_path))

    return validate_links(links, base_dir, file_path)

def collect_markdown_files(path_pattern):
    """
    Collects markdown files based on the provided path pattern.
    Supports:
    - Single file path
    - Directory path (finds all .md files recursively)
    - Glob patterns (e.g., *.md, docs/**/*.md)
    """
    files = []
    path = Path(path_pattern)
    
    # Check if it's an exact file match
    if path.is_file():
        files.append(str(path))
    # Check if it's a directory
    elif path.is_dir():
        # Find all .md files recursively
        files.extend([str(f) for f in path.rglob('*.md')])
    else:
        # Try glob pattern matching
        matched_files = glob.glob(path_pattern, recursive=True)
        # Filter to only markdown files
        files.extend([f for f in matched_files if f.endswith('.md') and os.path.isfile(f)])
    
    return sorted(files)

# Example Usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate all links in Markdown file(s).",
        epilog="Examples:\n"
               "  %(prog)s README.md                    # Single file\n"
               "  %(prog)s docs/                        # All .md files in directory\n"
               "  %(prog)s '*.md'                       # All .md files in current dir\n"
               "  %(prog)s 'docs/**/*.md'               # All .md files recursively\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path to Markdown file, directory, or glob pattern")
    args = parser.parse_args()
    
    # Collect all files to process
    files_to_check = collect_markdown_files(args.path)
    
    if not files_to_check:
        print(f"No markdown files found matching: {args.path}")
        sys.exit(1)
    
    print(f"Found {len(files_to_check)} markdown file(s) to check.\n")
    
    # Process all files and track results
    results = {}
    for file_path in files_to_check:
        print(f"\n{'='*80}")
        print(f"Checking: {file_path}")
        print('='*80)
        results[file_path] = validate_markdown_links(file_path)
    
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
