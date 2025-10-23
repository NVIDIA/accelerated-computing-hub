#!/usr/bin/env python3
"""
Process syllabus files and create modified docker-compose files.

For each syllabus file in tutorials/*/notebooks/syllabi/*.ipynb, this script:
1. Copies the corresponding tutorials/*/brev/docker-compose.yml file
2. Modifies it by setting the default-jupyter-url to the syllabus file path
3. Outputs the modified file to tutorials/*/notebooks/syllabi/{syllabi-name}__docker_compose.yml
"""

import re
from pathlib import Path


def extract_working_dir(content: str) -> str:
    """
    Extract the working-dir anchor value from docker-compose content.

    Args:
        content: Original docker-compose.yml content

    Returns:
        The working directory path, or empty string if not found
    """
    match = re.search(
        r'working-dir:\s*&working-dir\s+(.+?)(?:\s*$)',
        content,
        flags=re.MULTILINE
    )
    if match:
        return match.group(1).strip()
    return ""


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
        f'\\1 ["{jupyter_url}"]',
        content,
        flags=re.MULTILINE
    )
    return modified


def main():
    """Main processing function."""
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

        # Read the docker-compose file as text (to preserve YAML anchors)
        with open(docker_compose_src, 'r') as f:
            compose_content = f.read()

        # Extract the working-dir anchor value
        working_dir = extract_working_dir(compose_content)
        if not working_dir:
            print(f"⚠️  Warning: Could not extract working-dir from {docker_compose_src}")
            continue

        # Process each syllabi file
        for syllabi_file in syllabi_dir.glob('*.ipynb'):
            # Calculate the relative path from working-dir to syllabi file
            # The working_dir in docker-compose is /accelerated-computing-hub/tutorials/{tutorial}/notebooks
            # The syllabi file is at tutorials/{tutorial}/notebooks/syllabi/{file}.ipynb
            # So the relative path is: syllabi/{file}.ipynb
            # JupyterLab requires the "lab/tree/" prefix to open notebooks
            # We also append ?file-browser-path={working-dir} to set the file browser location
            syllabi_relative_path = f"/lab/tree/syllabi/{syllabi_file.name}?file-browser-path={working_dir}"

            print(f"  ✓ Processing: {syllabi_file.name}")
            print(f"    Jupyter URL: {syllabi_relative_path}")

            # Modify the default-jupyter-url anchor
            modified_content = modify_docker_compose(compose_content, syllabi_relative_path)

            # Write modified docker-compose file next to the syllabi file
            # Pattern: /path/to/X.ipynb -> /path/to/X__docker_compose.yml
            syllabi_name = syllabi_file.stem # Filename without extension
            output_file = syllabi_dir / f"{syllabi_name}__docker_compose.yml"
            with open(output_file, 'w') as f:
                f.write(modified_content)

            processed_files.append({
                'tutorial': tutorial_name,
                'syllabi': syllabi_name,
                'path': str(output_file)
            })

    print(f"\n✅ Successfully processed {len(processed_files)} syllabi files")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
