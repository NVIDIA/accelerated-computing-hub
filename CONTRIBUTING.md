# Contributing

Please use the following guidelines when contributing to this project.

Before contributing signficicant changes, please begin a discussion of the
desired changes via a GitHub Issue to prevent doing unnecessary or overlapping
work.

## Development Scripts

You can use the following scripts to help with development:

* **`brev/dev-build.bash [<tutorial-name>]`** - Builds Docker containers for tutorials. If a tutorial name is provided (e.g., `accelerated-python`), builds only that tutorial; if no argument is provided, builds all tutorials. Automatically generates Dockerfiles from HPCCM recipes if present.

* **`brev/dev-start.bash <tutorial-name>`** - Starts Docker containers for a tutorial, mounting your local repository as `/accelerated-computing-hub` in the containers.

* **`brev/dev-stop.bash <tutorial-name>`** - Stops Docker containers for a tutorial.

* **`brev/dev-shell.bash <tutorial-name|docker-compose-file> <service-name>`** - Starts an interactive bash shell in a Docker container for a tutorial. Can accept either a tutorial name or a path to docker-compose.yml file, plus the service name (e.g., `base`, `jupyter`, `nsight`).

* **`brev/dev-test.bash <tutorial-name|docker-compose-file>`** - Tests a Docker Compose file with the local repository mounted. Calls `test-docker-compose.bash` to run the tutorial's tests.

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

The preferred license for contributions to this project is the detailed in the
[LICENSE file](LICENSE).

Contributions must include a "signed off by" tag in the commit message for the
contributions asserting the signing of the developers certificate of origin
(https://developercertificate.org/). A GPG-signed commit with the "signed off
by" tag is preferred.

## Styling

Please use the guidelines detailed in the [notebook template](docs/notebook_template.ipynb) for contributions.

## Contributing Labs/Modules

A module should have the following directory structure:

* The base of the module should contain a README.ipynb file with a brief
  introduction to the module and links to the individual labs for each
  language translation and programming language available.
* The base of the module should contain a subdirectory for each programming language if applicable. Each
  of these directories should contain a directory for each language
  translation provided (English, for instance).
* The base of the module should contain an `images` directory that contains
  images that will be used in common between multiple notebooks.
* For each programming language and translation there should be a file named
  `README.ipynb` containing the actual lab instructions. A single file name
  is used in this way to simplify finding the starting point for each new
  lab.
* Each lab translation and programming language combination should have a
  `solutions` directory containing correct solutions.

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
