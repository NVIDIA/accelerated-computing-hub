#!/usr/bin/env python3
"""
Process syllabus files and create modified docker-compose files.

For each syllabus file in tutorials/*/notebooks/syllabi/*.ipynb, this script:
1. Copies the corresponding tutorials/*/brev/docker-compose.yml file
2. Modifies it by setting the default-jupyter-url to the syllabus file path
3. Outputs the modified file to tutorials/*/notebooks/syllabi/{syllabus-name}__docker_compose.yml
4. Creates a syllabi.md file linking to all syllabi and their docker-compose files
"""

import re
from pathlib import Path
from typing import List, Tuple


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
        r'(default-jupyter-url:\s*&default-jupyter-url)(\s+\[.*?\])?(\s*)$',
        f'\\1 ["{jupyter_url}"]',
        content,
        flags=re.MULTILINE
    )
    return modified


def generate_syllabi_markdown(tutorial_name: str, syllabi_info: List[Tuple[str, str]]) -> str:
    """
    Generate a markdown file content listing all syllabi and their docker-compose files.

    Args:
        tutorial_name: Name of the tutorial
        syllabi_info: List of tuples (syllabus_filename, docker_compose_filename)

    Returns:
        Generated markdown content
    """
    lines = [
        f"# {tutorial_name.replace('-', ' ').title()} - Syllabi\n",
        "\nThis directory contains different syllabi for this tutorial. Each syllabus is described in a notebook and has an associated Docker Compose configuration file that can be used to create a Brev Launchable.\n",
        "\n## Available Syllabi\n"
    ]

    for syllabus_file, docker_compose_file in sorted(syllabi_info):
        # Create a human-readable title from the filename
        title = syllabus_file.replace('.ipynb', '').replace('_', ' ').replace('  ', ' - ')

        lines.append(f"\n### {title}\n")
        lines.append(f"- **Notebook**: [{syllabus_file}]({syllabus_file})\n")
        lines.append(f"- **Docker Compose**: [{docker_compose_file}]({docker_compose_file})\n")

    return ''.join(lines)


def main():
    """Main processing function."""
    # Find all tutorials directories
    tutorials_base = Path('tutorials')

    if not tutorials_base.exists():
        print(f"❌ Error: {tutorials_base} directory not found")
        return 1

    processed_count = 0

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

        # Track syllabi info for generating the markdown file
        syllabi_info = []

        # Process each syllabus file
        for syllabus_file in syllabi_dir.glob('*.ipynb'):
            # JupyterLab requires the "lab/tree/" prefix to open notebooks
            # We also append ?file-browser-path={working-dir} to set the file browser location
            syllabus_url = f"/lab/tree{working_dir}/syllabi/{syllabus_file.name}?file-browser-path={working_dir}"

            print(f"  ✓ Processing: {syllabus_file.name}")
            print(f"    Jupyter URL: {syllabus_url}")

            # Modify the default-jupyter-url anchor
            modified_content = modify_docker_compose(compose_content, syllabus_url)

            # Write modified docker-compose file next to the syllabus file
            # Pattern: /path/to/X.ipynb -> /path/to/X__docker_compose.yml
            syllabus_name = syllabus_file.stem # Filename without extension
            docker_compose_filename = f"{syllabus_name}__docker_compose.yml"
            output_file = syllabi_dir / docker_compose_filename
            with open(output_file, 'w') as f:
                f.write(modified_content)

            # Track this syllabus for the markdown file
            syllabi_info.append((syllabus_file.name, docker_compose_filename))

            processed_count += 1

        # Generate syllabi.md file if there are any syllabi
        if syllabi_info:
            markdown_content = generate_syllabi_markdown(tutorial_name, syllabi_info)
            markdown_file = syllabi_dir / 'syllabi.md'
            with open(markdown_file, 'w') as f:
                f.write(markdown_content)
            print(f"  📝 Generated: syllabi.md")

    print(f"\n✅ Successfully processed {processed_count} syllabi files")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
