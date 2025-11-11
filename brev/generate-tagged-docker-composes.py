#!/usr/bin/env python3
"""
Generate docker-compose files with specific image tags.

This script creates modified versions of docker-compose files with updated image tags
for specific commits or branches. It can also generate syllabi versions.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


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


def modify_image_tag(content: str, new_tag: str) -> str:
    """
    Modify docker-compose content by updating the image tag.

    Args:
        content: Original docker-compose.yml content
        new_tag: New image tag to use (e.g., "main-latest", "main-git-abc1234")

    Returns:
        Modified docker-compose.yml content
    """
    # Pattern matches: image: &image ghcr.io/owner/repo:old-tag
    # and replaces with: image: &image ghcr.io/owner/repo:new-tag
    modified = re.sub(
        r'(image:\s*&image\s+[^:\s]+):([^\s]+)',
        f'\\1:{new_tag}',
        content,
        flags=re.MULTILINE
    )
    return modified


def modify_jupyter_url(content: str, jupyter_url: str) -> str:
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


def generate_base_composes(tutorial_dir: Path, image_tag: str, output_dir: Path) -> int:
    """
    Generate modified base docker-compose file with the specified image tag.

    Args:
        tutorial_dir: Path to the tutorial directory
        image_tag: Image tag to use
        output_dir: Directory to write the modified file

    Returns:
        Number of files generated
    """
    docker_compose_src = tutorial_dir / 'brev' / 'docker-compose.yml'
    if not docker_compose_src.exists():
        print(f"⚠️  Warning: No docker-compose.yml found for {tutorial_dir.name}")
        return 0

    with open(docker_compose_src, 'r') as f:
        compose_content = f.read()

    # Modify the image tag
    modified_content = modify_image_tag(compose_content, image_tag)

    # Create output directory structure
    tutorial_name = tutorial_dir.name
    output_tutorial_dir = output_dir / tutorial_name / 'brev'
    output_tutorial_dir.mkdir(parents=True, exist_ok=True)

    # Write modified docker-compose file
    output_file = output_tutorial_dir / 'docker-compose.yml'
    with open(output_file, 'w') as f:
        f.write(modified_content)

    print(f"  ✓ Generated: {output_file}")
    return 1


def generate_syllabi_composes(
    tutorial_dir: Path,
    image_tag: str,
    output_dir: Path
) -> int:
    """
    Generate modified syllabi docker-compose files with the specified image tag.

    Args:
        tutorial_dir: Path to the tutorial directory
        image_tag: Image tag to use
        output_dir: Directory to write the modified files

    Returns:
        Number of files generated
    """
    # Check for syllabi directory
    syllabi_dir = tutorial_dir / 'notebooks' / 'syllabi'
    if not syllabi_dir.exists():
        return 0

    # Check for docker-compose.yml
    docker_compose_src = tutorial_dir / 'brev' / 'docker-compose.yml'
    if not docker_compose_src.exists():
        print(f"⚠️  Warning: No docker-compose.yml found for {tutorial_dir.name}")
        return 0

    tutorial_name = tutorial_dir.name

    # Read the docker-compose file as text (to preserve YAML anchors)
    with open(docker_compose_src, 'r') as f:
        compose_content = f.read()

    # Modify the image tag
    compose_content = modify_image_tag(compose_content, image_tag)

    # Extract the working-dir anchor value
    working_dir = extract_working_dir(compose_content)
    if not working_dir:
        print(f"⚠️  Warning: Could not extract working-dir from {docker_compose_src}")
        return 0

    count = 0

    # Process each syllabus file
    for syllabus_file in syllabi_dir.glob('*.ipynb'):
        # JupyterLab requires the "lab/tree/" prefix to open notebooks
        # We also append ?file-browser-path={working-dir} to set the file browser location
        syllabus_url = f"/lab/tree{working_dir}/syllabi/{syllabus_file.name}?file-browser-path={working_dir}"

        # Modify the default-jupyter-url anchor
        modified_content = modify_jupyter_url(compose_content, syllabus_url)

        # Create output directory structure
        output_syllabi_dir = output_dir / tutorial_name / 'notebooks' / 'syllabi'
        output_syllabi_dir.mkdir(parents=True, exist_ok=True)

        # Write modified docker-compose file next to the syllabus file
        # Pattern: X.ipynb -> X__docker_compose.yml
        syllabus_name = syllabus_file.stem  # Filename without extension
        output_file = output_syllabi_dir / f"{syllabus_name}__docker_compose.yml"
        with open(output_file, 'w') as f:
            f.write(modified_content)

        print(f"  ✓ Generated: {output_file}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description='Generate docker-compose files with specific image tags'
    )
    parser.add_argument(
        '--image-tag',
        required=True,
        help='Image tag to use (e.g., "main-latest", "main-git-abc1234")'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--tutorial',
        type=str,
        help='Specific tutorial to process (processes all if not specified)'
    )
    parser.add_argument(
        '--type',
        choices=['base', 'syllabi', 'all'],
        default='all',
        help='Type of docker-compose files to generate'
    )

    args = parser.parse_args()

    # Find tutorials directories
    tutorials_base = Path('tutorials')

    if not tutorials_base.exists():
        print(f"❌ Error: {tutorials_base} directory not found")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_count = 0
    syllabi_count = 0

    # Determine which tutorials to process
    if args.tutorial:
        tutorial_dirs = [tutorials_base / args.tutorial]
    else:
        tutorial_dirs = [d for d in tutorials_base.iterdir() if d.is_dir()]

    print(f"\n🏷️  Generating docker-compose files with tag: {args.image_tag}")
    print(f"📁 Output directory: {args.output_dir}\n")

    for tutorial_dir in tutorial_dirs:
        if not tutorial_dir.is_dir():
            continue

        tutorial_name = tutorial_dir.name
        print(f"📦 Processing tutorial: {tutorial_name}")

        # Generate base docker-compose files
        if args.type in ['base', 'all']:
            base_count += generate_base_composes(
                tutorial_dir,
                args.image_tag,
                args.output_dir
            )

        # Generate syllabi docker-compose files
        if args.type in ['syllabi', 'all']:
            syllabi_count += generate_syllabi_composes(
                tutorial_dir,
                args.image_tag,
                args.output_dir
            )

    print(f"\n✅ Successfully generated:")
    if args.type in ['base', 'all']:
        print(f"   • {base_count} base docker-compose file(s)")
    if args.type in ['syllabi', 'all']:
        print(f"   • {syllabi_count} syllabi docker-compose file(s)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
