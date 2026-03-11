# Contributing

Please use the following guidelines when contributing to this project.

Before contributing significant changes, please begin a discussion of the desired changes via a GitHub Issue to prevent doing unnecessary or overlapping work.

## Development Scripts

The `brev/` directory contains scripts for building, running, and testing tutorial Docker containers locally, as well as validation and CI helper scripts.

### Docker Workflow (`brev/dev-*`)

These are the primary scripts for local development. They all accept a tutorial name (the directory name under `tutorials/`, e.g. `accelerated-python`) and manage Docker Compose services for that tutorial.

* **`brev/dev-build.bash [<tutorial-name>]`** - Builds Docker images. If a tutorial name is provided, builds only that tutorial; if omitted, builds all tutorials discovered by `discover-tutorials.bash`. Automatically generates Dockerfiles from HPCCM recipes if present.

* **`brev/dev-start.bash [--mount|--no-mount] <tutorial-name>`** - Starts containers for a tutorial. By default (`--mount`), bind-mounts the local repository into the containers at `/accelerated-computing-hub` so edits are reflected immediately. Use `--no-mount` to run from the image content only. Rebinds ports to `0.0.0.0` for local access.

* **`brev/dev-stop.bash <tutorial-name>`** - Stops and tears down containers for a tutorial.

* **`brev/dev-shell.bash [--mount|--no-mount] <tutorial-name|docker-compose-file> <service-name>`** - Opens an interactive bash shell inside a running container. Accepts either a tutorial name or a path to a `docker-compose.yml` file, plus the service name (e.g. `base`, `jupyter`, `nsys`, `ncu`).

* **`brev/dev-test.bash [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]`** - Runs the tutorial's test suite inside Docker. Defaults to `--no-mount` (tests the image as-built). Extra arguments are forwarded to the tutorial's `test.bash` script (e.g. `brev/dev-test.bash accelerated-python test/test_notebooks.py -k "03"`).

### Validation Scripts (`brev/test-*`)

These scripts are used by the pre-commit hooks and CI. You can also run them directly.

* **`brev/test-notebook-format.py [<tutorial-name>] [--fix]`** - Checks that notebooks are in canonical format (1-space indentation, sorted keys, standard metadata, clean outputs for non-SOLUTION notebooks). Pass `--fix` to auto-rewrite notebooks to canonical form.

* **`brev/test-git-lfs.bash [<path>...]`** - Verifies that binary files (images, PDFs, presentations) are properly tracked by Git LFS and that no `.gitattributes` files exist in subdirectories.

* **`brev/test-git-signatures.bash [<base-ref>]`** - Checks that all commits on the current branch (since `origin/main` by default) are cryptographically signed.

* **`brev/test-links.bash <path>`** - Converts notebooks to markdown, then runs [lychee](https://github.com/lycheeverse/lychee) to check for broken links. Operates on a temporary copy to avoid polluting the working tree.

* **`brev/test-docker-compose.bash [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]`** - The lower-level test runner that `dev-test.bash` delegates to. Starts containers, checks for restart loops, runs the tutorial's tests, verifies restart behavior, and tears everything down.

### Other Utility Scripts

* **`brev/discover-tutorials.bash`** - Prints one line per tutorial directory (relative to repo root) for every tutorial that has a `brev/docker-compose.yml`. Used by `dev-build.bash` and CI.

* **`brev/generate-tagged-docker-composes.py`** - Generates Docker Compose files with specific image tags for a given branch/commit. Used by CI to produce versioned artifacts that are pushed to the `generated` branch.

## Pre-Commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to run automated checks before commits and pushes. Install the hooks after cloning:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

The following hooks are configured:

| Hook | Stage | Files | What it checks |
|---|---|---|---|
| **`notebook-format`** | pre-commit | `*.ipynb` | Validates that notebooks are in canonical format (consistent cell metadata, output ordering, etc.) to prevent noisy diffs from editor reformatting. |
| **`git-lfs`** | pre-commit | Binary files (`*.pptx`, `*.pdf`, `*.png`, `*.jpg`, etc.) | Ensures binary files are tracked by Git LFS rather than committed directly to the repository. |
| **`git-signatures`** | pre-push | All commits | Verifies that all commits being pushed have valid cryptographic signatures (see [Signing Your Commits](#signing-your-commits)). |
| **`test-links`** | manual | `*.md`, `*.ipynb` | Checks for broken links in markdown and notebook files. Run manually with `pre-commit run test-links --all-files`. |

If a hook fails, fix the issue and re-stage your changes before committing again.

## Branches

### `main`

The primary integration branch. All changes land here via pull requests.

### `feature/<description>` and `fix/<description>`

Development branches for new features and bug fixes, respectively. Create these from `main` and open a pull request when ready.

- `feature/<description>` — new functionality or enhancements (e.g. `feature/matplotlib-dark-mode-detect`).
- `fix/<description>` — bug fixes or corrections (e.g. `fix/dockerfile-copy-chmod-layers`).

### `pull-request/<number>`

These branches are created to enable CI on pull requests from forks. Because this project uses NVIDIA's self-hosted GitHub Actions runners, workflows cannot run directly on fork PRs for security reasons. NVIDIA's [**copy-pr-bot**](https://docs.gha-runners.nvidia.com/cpr/vetters/) automates this process -- when a maintainer determines a fork PR is safe to test, they push a copy of the PR to a `pull-request/<number>` branch on the upstream repo (where `<number>` is the PR number). The CI workflow is configured to trigger on branches matching the `pull-request/[0-9]+` pattern, so pushing to this branch runs the full build and test pipeline. The results are reported back on the original PR because the commit SHAs match. See the [Reviewing PRs From Forks](#reviewing-prs-from-forks) section below for the step-by-step process.

### `generated`

A CI-managed branch that stores generated artifacts (e.g. tagged Docker Compose files). The build workflow automatically commits updated Docker Compose files to this branch under `<source-branch>/tutorials/<tutorial-name>/` after each successful build. You should never need to commit to this branch manually.

### Event branches (`YYYY-MM-<event-name>`)

Long-lived branches for conference workshops and training events (e.g. `2025-09-ndc-techtown`, `2026-01-internal-cuda-tile-hackathon`). These capture the exact state of tutorials as delivered at a specific event.

## Commit Messages

Commit messages use a **`Category[/Subcategory]: Summary.`** format. The summary is a sentence in imperative mood that starts with a capital letter and ends with a period.

### Format

```
Category[/Subcategory[/...]]: Imperative summary of the change.
```

Examples:

```
Docker: Fix entrypoint failure when target UID already exists.
Tutorials/Accelerated Python/Kernel Authoring: Add correctness check cell before profiling copy_blocked kernel.
CI: Add nightly schedule trigger to build workflow.
```

### Categories

Use the most specific category path that accurately describes what was changed. The top-level category identifies the broad area of the codebase, and subcategories narrow it down.

| Category | When to use |
|---|---|
| **`Tutorials/<tutorial>`** | Changes to tutorial content (notebooks, scripts, solutions). Use subcategories to identify the specific module when applicable, e.g. `Tutorials/Accelerated Python/Memory Spaces`, `Tutorials/CUDA Tile/Vector Add`. |
| **`Docker`** | Changes to Dockerfiles, Docker Compose files, entrypoint scripts, or container configuration that affect multiple tutorials or the shared Docker infrastructure. |
| **`Brev`** | Changes specific to the Brev deployment configuration or Brev-specific entrypoint behavior. Use subcategories for specific services, e.g. `Brev/Nsight`, `Brev/Entrypoint`. |
| **`CI`** | Changes to GitHub Actions workflows, CI scripts, or CI configuration. |
| **`Scripts`** | Changes to development and utility scripts (e.g. `brev/dev-build.bash`). |
| **`Tests`** | Changes to test infrastructure, test scripts, or test validation. |
| **`Nsight`** | Changes to Nsight Systems/Compute integration, Nsight Streamer, or profiling configuration. |
| **`Docs`** | Changes to documentation files (READMEs, contributing guides, link fixes). |
| **`Slides`** | Changes to presentation materials. |

### Summary Guidelines

- Use **imperative mood**: "Fix bug" not "Fixed bug" or "Fixes bug".
- Start with a **capital letter** and end with a **period**.
- Describe the **what and why**, not the how: "Fix entrypoint failure when target UID already exists." rather than "Add an if-check around useradd."
- If the commit fixes a GitHub issue, append `Fixes #<number>.` to the end of the summary.

## Pull Requests From Forks

When opening a pull request from a fork, please check the **"Allow edits from maintainers"** checkbox. This lets maintainers push small fixes (typos, formatting, CI tweaks) directly to your PR branch without requiring a round-trip, which significantly speeds up the review process. You can find this checkbox at the bottom of the "Open a pull request" form on GitHub. See the [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork) for more details.

## Pre-Commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to run automated
checks before commits and pushes. Install the hooks after cloning:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

The following hooks are configured:

### `notebook-format` (pre-commit)

Checks that Jupyter notebooks are in **canonical format**: 1-space JSON
indentation, sorted keys, standard metadata (kernelspec, colab settings,
language_info), nbformat 4.5, and cell IDs present. Non-SOLUTION notebooks
must have clean outputs (no outputs or execution counts). To auto-fix
formatting issues:

```bash
python3 brev/test-notebook-format.py --fix
```

### `git-lfs` (pre-commit)

Checks that binary files (`.pptx`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.gif`,
`.webp`) are properly tracked by Git LFS. Also verifies that no
`.gitattributes` files exist in subdirectories. If a file isn't tracked, fix it
with:

```bash
git rm --cached <file>
git add <file>
```

### `git-signatures` (pre-push)

Checks that all commits on your branch (compared to `origin/main`) are
cryptographically signed. Runs on `git push`, not on commit. See
[GitHub's documentation on commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification)
for setup instructions.

### `test-links` (manual)

Checks that links in Markdown files and Jupyter notebooks are valid using
[lychee](https://github.com/lycheeverse/lychee). This hook is configured as
a **manual** stage and won't run automatically. Run it explicitly with:

```bash
pre-commit run test-links --hook-stage manual --all-files
```

## License

The preferred license for contributions to this project is detailed in the [LICENSE file](LICENSE).

## Signing Your Commits

This project requires that all commits and contributions must be **verified** on GitHub. This is necessary to protect against security threats and protect both our codebase and our CI system.

This means each commit needs two things:

1. A **`Signed-off-by` trailer** asserting the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).
2. A **cryptographic signature** (GPG or SSH) so GitHub displays the "Verified" badge.

### Signed-off-by (DCO)

The `Signed-off-by` trailer certifies that you wrote the contribution or otherwise have the right to submit it under the project's license. Add it by passing `--signoff` (or `-s`) when committing:

```bash
git commit -s -m "CI: Add nightly schedule trigger to build workflow."
```

This appends a line like the following to the commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

Your name and email must match your Git identity (`user.name` and `user.email`).

### CryptographicG Commit Signing

We require that all commits to be cryptographically signed so they display as **"Verified"** on GitHub. The `git-signatures` pre-commit hook and CI both enforce this. You can sign with either a GPG key or an SSH key.

#### Option A: GPG Signing

1. [Generate a GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key) and [add it to your GitHub account](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account).
2. Tell Git to use your key:

```bash
gpg --list-secret-keys --keyid-format=long
git config --global user.signingkey <YOUR_KEY_ID>
```

3. Enable automatic signing so you don't have to pass `-S` every time:

```bash
git config --global commit.gpgsign true
```

#### Option B: SSH Signing

1. [Add your SSH key to your GitHub account as a **signing** key](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key#telling-git-about-your-ssh-key).
2. Configure Git to use SSH for signing:

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

### Recommended Git Configuration

To sign and sign-off on every commit by default, add both settings:

```bash
git config --global commit.gpgsign true
git config --global format.signoff true
```

With these defaults, a simple `git commit -m "message"` will produce a signed, signed-off commit.

## Styling

Please use the guidelines detailed in the [notebook template](docs/notebook_template.ipynb) for contributions.

## Tutorial Architecture

For details on the tutorial directory structure, Docker Compose services, notebook conventions (exercise vs. solution notebooks), tests, and deployment infrastructure, see the [Brev Launchable Architecture](docs/brev_launchable_architecture.md) documentation.

## Reviewing PRs From Forks

Due to security challenges of running CI on self-hosted GitHub Actions runners workflows on PRs from forks will not trigger automatically.

To work around this a maintainer needs to inspect the PR to ensure there are no concerns with running it, then push a copy of the PR to a branch on the upstream repo. This will trigger the CI to run. The results of the workflow will be reported back on the PR due to the matching SHAs of each commit.

```bash
# Use the GitHub CLI to check out the PR
gh pr checkout 123

# Push to a PR branch on the upstream
git push upstream pull-request/123
```

If the contributor makes further changes these will also need to be pulled/pushed to trigger the CI again.

```bash
# Check out the PR branch again
gh pr checkout 123
# Or just `git checkout <name of branch> && git pull` if you already have it

# Push to upstream PR branch
git push upstream pull-request/123  # You may need --force id the contributor has rewritten their history
```

See the [GitHub Actions NVIDIA Runners documentation for more information](https://docs.gha-runners.nvidia.com/platform/onboarding/pull-request-testing/).

## Attribution

Portions adopted from [https://github.com/OpenACC/openacc-training-materials/blob/master/CONTRIBUTING.md](https://github.com/OpenACC/openacc-training-materials/blob/master/CONTRIBUTING.md)
