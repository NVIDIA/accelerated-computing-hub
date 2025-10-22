#!/usr/bin/env python3
"""
Process syllabi files and create modified docker-compose files for GitHub Pages.

For each syllabi file in tutorials/*/notebooks/syllabi/*.ipynb, this script:
1. Copies the corresponding tutorials/*/brev/docker-compose.yml file
2. Modifies it by setting the default-jupyter-url to the syllabi file path
3. Outputs the modified file to _site/{tutorial}/{syllabi-name}/docker-compose.yml
4. Creates an index.html listing all processed files
"""

import re
from pathlib import Path
from collections import defaultdict


def modify_docker_compose(content: str, jupyter_url: str) -> str:
    """
    Modify docker-compose content by setting the default-jupyter-url anchor.

    Args:
        content: Original docker-compose.yml content
        jupyter_url: Path to set as the default Jupyter URL

    Returns:
        Modified docker-compose.yml content
    """
    modified = re.sub(
        r'(default-jupyter-url:\s*&default-jupyter-url)\s*$',
        f'\\1 {jupyter_url}',
        content,
        flags=re.MULTILINE
    )
    return modified


def generate_index_html(processed_files: list) -> str:
    """
    Generate an HTML index page listing all processed docker-compose files.

    Args:
        processed_files: List of dicts with 'tutorial', 'syllabi', and 'path' keys

    Returns:
        HTML content as a string
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syllabi Docker Compose Files</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #76B900;
            padding-bottom: 0.5rem;
        }
        h2 {
            color: #555;
            margin-top: 2rem;
        }
        .file-list {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file-item {
            padding: 1rem;
            margin: 0.5rem 0;
            background: #f9f9f9;
            border-radius: 4px;
            border-left: 4px solid #76B900;
        }
        .file-item a {
            color: #0066cc;
            text-decoration: none;
            font-weight: 500;
        }
        .file-item a:hover {
            text-decoration: underline;
        }
        .tutorial-name {
            color: #888;
            font-size: 0.9em;
            margin-top: 0.25rem;
        }
        code {
            background: #eee;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🚀 Syllabi Docker Compose Files</h1>
    <p>Modified Docker Compose files with pre-configured syllabi for each tutorial.</p>
"""

    # Group by tutorial
    by_tutorial = defaultdict(list)
    for item in processed_files:
        by_tutorial[item['tutorial']].append(item)

    for tutorial in sorted(by_tutorial.keys()):
        tutorial_title = tutorial.replace('-', ' ').title()
        html += f"\n<h2>{tutorial_title}</h2>\n<div class='file-list'>\n"

        for item in sorted(by_tutorial[tutorial], key=lambda x: x['syllabi']):
            syllabi_display = item['syllabi'].replace('_', ' ').replace('__', ' - ')
            html += f"""    <div class='file-item'>
        <a href="{item['path']}" download>{syllabi_display}</a>
        <div class='tutorial-name'><code>{item['path']}</code></div>
    </div>
"""
        html += "</div>\n"

    html += """
    <hr style="margin-top: 3rem; border: none; border-top: 1px solid #ddd;">
    <p style="color: #888; font-size: 0.9em;">
        Generated from the <a href="https://github.com/nvidia/accelerated-computing-hub">Accelerated Computing Hub</a> repository.
    </p>
</body>
</html>
"""
    return html


def main():
    """Main processing function."""
    # Create output directory
    output_dir = Path('_site')
    output_dir.mkdir(exist_ok=True)

    # Find all tutorials directories
    tutorials_base = Path('tutorials')

    if not tutorials_base.exists():
        print(f"❌ Error: {tutorials_base} directory not found")
        return 1

    processed_files = []

    for tutorial_dir in tutorials_base.iterdir():
        if not tutorial_dir.is_dir():
            continue

        # Check for syllabi directory
        syllabi_dir = tutorial_dir / 'notebooks' / 'syllabi'
        if not syllabi_dir.exists():
            continue

        # Check for docker-compose.yml
        docker_compose_src = tutorial_dir / 'brev' / 'docker-compose.yml'
        if not docker_compose_src.exists():
            print(f"⚠️  Warning: No docker-compose.yml found for {tutorial_dir.name}")
            continue

        tutorial_name = tutorial_dir.name
        print(f"\n📦 Processing tutorial: {tutorial_name}")

        # Process each syllabi file
        for syllabi_file in syllabi_dir.glob('*.ipynb'):
            # Read the docker-compose file as text (to preserve YAML anchors)
            with open(docker_compose_src, 'r') as f:
                compose_content = f.read()

            # Calculate the relative path from working-dir to syllabi file
            # The working_dir in docker-compose is /accelerated-computing-hub/tutorials/{tutorial}/notebooks
            # The syllabi file is at tutorials/{tutorial}/notebooks/syllabi/{file}.ipynb
            # So the relative path is: syllabi/{file}.ipynb
            # JupyterLab requires the "lab/tree/" prefix to open notebooks
            syllabi_relative_path = f"lab/tree/syllabi/{syllabi_file.name}"

            print(f"  ✓ Processing: {syllabi_file.name}")
            print(f"    Jupyter URL: {syllabi_relative_path}")

            # Modify the default-jupyter-url anchor
            modified_content = modify_docker_compose(compose_content, syllabi_relative_path)

            # Create output directory structure
            syllabi_name = syllabi_file.stem  # filename without extension
            output_subdir = output_dir / tutorial_name / syllabi_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Write modified docker-compose file
            output_file = output_subdir / 'docker-compose.yml'
            with open(output_file, 'w') as f:
                f.write(modified_content)

            processed_files.append({
                'tutorial': tutorial_name,
                'syllabi': syllabi_name,
                'path': f"{tutorial_name}/{syllabi_name}/docker-compose.yml"
            })

    # Create an index.html file
    html_content = generate_index_html(processed_files)
    with open(output_dir / 'index.html', 'w') as f:
        f.write(html_content)

    print(f"\n✅ Successfully processed {len(processed_files)} syllabi files")
    print(f"📁 Output directory: {output_dir.absolute()}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
