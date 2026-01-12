# Contributing to the CUDA C++ Tutorial

Thank you for your interest in contributing to the **Fundamentals of Accelerated Computing with Modern CUDA C++** course! 
This guide provides the context, philosophy, and practical instructions you need to create content that aligns with the course's goals and style.

## Course Philosophy

This is a **top-down**, **productivity-focused** introductory CUDA course. 
We introduce high-level tools and concepts before diving into kernel programming, 
equipping learners with practical skills to leverage GPU acceleration effectively.

### Why Top-Down?

Traditional CUDA courses typically starts with the vector add kernel, which requires introducing many concepts simultaneously: 
execution space, memory space, asynchrony, parallel vs. sequential computing, and the CUDA thread hierarchy. 
This approach can overwhelm beginners and doesn't reflect modern CUDA C++ best practices.

We address this by introducing **one concept at a time**, building understanding incrementally.

### Why Productivity-Focused?

Traditional CUDA courses teach *how* to write kernels, so when faced with production tasks like sorting or matrix multiplication, the instinct is to write more kernels. 
But that's usually the wrong answer because accelerated libraries already provide speed-of-light implementations for common operations. 
This course teaches *when* to write custom CUDA code, and more importantly, when not to.

### Module Organization

The course is divided into **modules**, each designed to be completable in a single session (up to ~2 hours) for an in-person delivery such as lectures or workshops:

**Module 01 — CUDA Made Easy: Accelerating Applications with Parallel Algorithms**
- Execution spaces 
- Vocabulary types 
- Serial vs parallel algorithms
- Memory spaces 

**Module 02 — Unlocking the GPU’s Full Potential: Asynchrony and CUDA Streams**
- Synchronous vs asynchronous execution
- CUDA streams and concurrency
- Pinned memory
- Profiling with Nsight Systems

**Module 03 — Implementing New Algorithms with CUDA Kernels**
- Thread hierarchy
- Kernel programming
- Atomics and synchronization
- Shared memory
- Cooperative algorithms


### Guiding Principles

When creating content, keep these principles in mind:

1. **Incremental Learning**: Introduce **one concept at a time** to avoid overwhelming learners. Each notebook should have a clear, focused objective.

2. **Practical Productivity**: Emphasize **practical applications** that enhance developer productivity and real-world performance. Industry relevance over academic exercises.

3. **General-Purpose Education**: Teach general-purpose CUDA accelerated computing, **not domain-specific applications**. Examples should be accessible to developers from any background.

---

## Target Audience

This course targets developers with:
- **Basic familiarity with C++11** (lambdas, range-based for loops, standard containers)
- **No prior experience** with parallel computing or CUDA
- **Industry focus**: seeking to leverage GPU acceleration for practical applications

**Do not assume** familiarity with:
- GPU architecture details
- Parallel programming concepts (threads, synchronization, etc.)
- Domain-specific knowledge (unless introducing the concept inline)

---

## Course Structure

### Module Organization

The course is organized into **modules**, each focusing on a cohesive set of concepts:

```
tutorials/cuda-cpp/notebooks/
├── 01.01-Introduction/          # Module 01: Parallel Algorithms
├── 01.02-Execution-Spaces/
├── ...
├── 02.01-Introduction/          # Module 02: Asynchrony & Streams
├── 02.02-Asynchrony/
├── ...
```

Each module begins with an **Introduction** notebook that:
- Sets the stage for what will be covered
- Lists prerequisites (concepts from previous modules)
- Defines measurable learning objectives
- Provides a table of contents linking to all notebooks in the module

### Naming Conventions

**Directories** follow the pattern:
```
XX.YY-Topic-Name/
```
Where:
- `XX` = Module number (01, 02, 03, ...)
- `YY` = Section number within module (01, 02, 03, ...)
- `Topic-Name` = Descriptive name using Title-Case-With-Dashes

**Notebooks** follow the pattern:
```
XX.YY.ZZ-Notebook-Name.ipynb
```
Where:
- `XX.YY` = Matches the parent directory
- `ZZ` = Sequence number within the section (01, 02, 03, ...)
- `Notebook-Name` = Descriptive name

**Examples:**
- Lesson: `01.02.01-Execution-Spaces.ipynb`
- Exercise: `01.02.02-Exercise-Annotate-Execution-Spaces.ipynb`
- Exercise: `01.02.03-Exercise-Changing-Execution-Space.ipynb`

### Directory Layout

Each section directory contains:

```
XX.YY-Topic-Name/
├── XX.YY.01-Topic-Name.ipynb          # Main lesson notebook
├── XX.YY.02-Exercise-Name.ipynb       # Exercise notebooks
├── XX.YY.03-Exercise-Another.ipynb
├── Images/                            # Diagrams and figures
│   ├── diagram.svg                    # Exported images
│   └── diagram.excalidraw             # Source files (Excalidraw)
├── Sources/                           # Starter code for exercises
│   ├── ach.h                          # Helper header (copy from course root)
│   └── exercise-starter.cpp
└── Solutions/                         # Complete solutions
    ├── ach.h
    └── exercise-solution.cpp
```

---

## Content Guidelines

### Notebook Structure

Each **lesson notebook** should follow this structure:

```markdown
# Topic Title

## Content

* [Section 1](#Section-1)
* [Section 2](#Section-2)
* [Exercise: Exercise Name](XX.YY.ZZ-Exercise-Name.ipynb)

---

Brief introduction setting context and motivation.
```

1. **Title and Table of Contents**: Start with a clear title and link to all sections and exercises
2. **Introduction**: Brief context connecting to prior material
3. **Conceptual Sections**: Teach concepts incrementally with code examples
4. **Conclusion**: Summarize and link to exercises/next notebook

Each **exercise notebook** should follow this structure:

```markdown
## Exercise: Exercise Name

Description of what the learner should accomplish.
Clear instructions on what to change (e.g., "Replace all `???` with...").
```

1. **Exercise description**: Clear statement of the goal
2. **Setup cell**: Download sources if running on Colab
3. **Starter code**: Code with `TODO` comments marking what to change
4. **Verification cell**: Compile and run to check results
5. **Hints**: Collapsible section with guidance
6. **Solution**: Collapsible section with explanation and solution code
7. **Next steps**: Link to the next notebook

### Writing Style

- **Be conversational**: Write as if explaining to a colleague
- **Use active voice**: "We update the temperature" not "The temperature is updated"
- **Explain the "why"**: Don't just show what to do — explain why it matters
- **Use inline code formatting** for: function names, variable names, file names, commands
- **Use emphasis** for: new terms, important concepts
- **Build intuition first**: Introduce concepts conceptually before showing code

**Example of writing style:**
```markdown
By default, code runs on the **host** side.
You are responsible for specifying which code should run on the **device**.
This should explain why using `nvcc` alone was insufficient: we haven't marked any code for execution on GPU.
```

### The Heat Simulation Example

Throughout the course, we use a **heat conduction simulation** as a running example. This example:
- Evolves in complexity as new concepts are introduced
- Provides a concrete and visual application
- Is general enough to avoid domain-specific knowledge

When adding new modules or sections, consider whether the heat simulation can be extended to demonstrate new concepts. 
If introducing a new example, make sure it:
- Is simple enough to understand quickly
- Demonstrates the concept clearly
- Doesn't require domain-specific knowledge

### Code Style

**C++ Code:**
- Use modern C++11/14 idioms (lambdas, `auto`, range-based for)
- Prefer algorithms over raw loops (`std::transform`, `thrust::transform`)
- Use clear and descriptive variable names
- Add comments explaining non-obvious code

**CUDA-specific:**
- Always annotate device-callable code with `__host__ __device__` when possible
- Use containers instead of raw memory allocations
- Use execution policies explicitly: `thrust::device`, `thrust::host`
- Prefer libraries when possible

**Example:**
```cpp
// Transform temperatures using Newton's law of cooling
auto transformation = [=] __host__ __device__ (float temp) { 
    return temp + k * (ambient_temp - temp); 
};
thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
```

**Compilation commands:**
```bash
# Standard compilation with extended lambdas
!nvcc --extended-lambda Sources/file.cpp -x cu -arch=native -o /tmp/a.out
!/tmp/a.out
```

### Exercises

Exercises are crucial for reinforcing concepts. Design exercises that:

1. **Have a clear objective**: What should the learner accomplish?
2. **Build incrementally**: Start simple, add complexity in later exercises
3. **Provide scaffolding**: Give starter code with clear `TODO` markers
4. **Are self-verifiable**: Output should clearly indicate success/failure

Here's an example (`01.02.02-Exercise-Annotate-Execution-Spaces`):

**Exercise description cell:**
```markdown
## Exercise: Annotate Execution Spaces

The notion of execution space is a foundational concept of accelerated computing. 
In this exercise you will verify your expectation of *where* any given code is executed.

Replace all `???` with `CPU` or `GPU`, based on where you think that specific line of code is executing.
The `ach::where_am_I` function is a helper function for you in this exercise.

After making all the changes, run the subsequent cell to verify your expectations.
```

**Starter code** (`Sources/no-magic-execution-space-changes.cpp`):
```cpp
#include "ach.h"

int main() {
  // TODO: Replace ??? with CPU or GPU
  ach::where_am_I("???");

  thrust::universal_vector<int> vec{1};
  thrust::for_each(thrust::device, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("???"); });

  thrust::for_each(thrust::host, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("???"); });

  ach::where_am_I("???");
}
```

The `ach::where_am_I` helper prints "Correct!" or "Wrong guess" — making success/failure immediately obvious.

**Hints cell** (collapsible, placed before solution):
```markdown
<details>
  <summary>Hints</summary>
  
  - For invocations in the main function consult the [Heterogeneous Programming Model](#) section
  - For invocations in lambdas consult the [Execution Policy](#) section
</details>
```

**Solution cell** (collapsible):
```markdown
<details>
  <summary>Solution</summary>

  Key points:
  - The main function always runs on the CPU
  - According to `thrust::device` execution policy, the first `thrust::for_each` invokes the lambda on the GPU
  - According to `thrust::host` execution policy, the second `thrust::for_each` invokes the lambda on the CPU

  Solution:
  ```c++
  ach::where_am_I("CPU");

  thrust::universal_vector<int> vec{1};
  thrust::for_each(thrust::device, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("GPU"); });

  thrust::for_each(thrust::host, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("CPU"); });

  ach::where_am_I("CPU");
  ```

  You can find the full solution [here](Solutions/no-magic-execution-space-changes.cpp).
</details>
```

### Images and Diagrams

- Store all images in the `Images/` subdirectory of a given notebook
- **Use SVG format** for diagrams when possible (scalable, editable)
- **Include source files**: We use [Excalidraw](https://excalidraw.com/) for diagrams. Include `.excalidraw` files alongside exported images for future maintenance.
- **Add alt text** for accessibility
- Use consistent visual style across the course

**Including images in notebooks:**
```markdown
![Heterogeneous programming model](Images/heterogeneous.png "Heterogeneous programming model")
```

---

## Technical Infrastructure

### Helper Library (`ach.h`)

When adding new helper functions, place them in the `ach` namespace inside `ach.h` header file in the directory of the notebook.
This keeps exercise code clean (`#include "ach.h"`) and provides a consistent interface across the course. 
When creating exercises, you can use these utilities or add new ones. 

### Environment Support

The course supports multiple execution environments:

1. **Google Colab**: Primary target for zero-setup experience
2. **NVIDIA Brev**: Cloud GPU development environment
3. **Local Docker**: For developers with local GPU access

**Colab compatibility**: Each notebook should include a setup cell that downloads necessary source files:
```python
import os

if os.getenv("COLAB_RELEASE_TAG"):  # If running in Google Colab:
    !mkdir -p Sources
    !wget https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/refs/heads/main/tutorials/cuda-cpp/notebooks/XX.YY-Topic/Sources/ach.h -nv -O Sources/ach.h
```

**Docker environment**: The `brev/` directory contains Docker configuration:
- `dockerfile`: Base image with CUDA toolkit and Jupyter
- `docker-compose.yml`: Service definitions for Jupyter and Nsight
- `requirements.txt`: Python dependencies

### Testing

Before submitting, verify your notebooks work in:

1. **Local environment**: Run all cells in order
2. **Google Colab**: Test the Colab workflow including source downloads

---

## Workflow

### Before You Start

1. **Open a GitHub Issue**: Discuss proposed changes before starting significant work
2. **Review existing content**: Understand the course flow and where your contribution fits
3. **Identify prerequisites**: What concepts must learners already know?
4. **Define learning objectives**: What will learners be able to do after completing your content?

### Development

1. **Fork and clone** the repository
2. **Create a feature branch**: `git checkout -b feature/module-XX-topic-name`
3. **Follow conventions**: Use the naming and structure patterns described above
4. **Test thoroughly**: Verify notebooks run cleanly from top to bottom
5. **Update README.md**: Add entries for new notebooks to the course index

### Submitting Changes

1. **Create a Pull Request**: Reference the related issue
2. **Address review feedback**: Iterate based on maintainer comments
3. **CI validation**: Ensure all automated checks pass

See the [main CONTRIBUTING.md](../../CONTRIBUTING.md) for repository-wide contribution guidelines.

---

## Questions?

If you have questions about contributing, please:
- Open a GitHub Issue for discussion
- Review existing notebooks for examples of style and structure
- Check the [YouTube playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr) for context on how content is delivered

Thank you for helping make CUDA education more accessible! 
