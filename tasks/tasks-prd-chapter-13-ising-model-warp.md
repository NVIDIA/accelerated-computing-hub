# Task List: Chapter 13 - Ising Model in Warp

## Relevant Files

- `Accelerated_Python_User_Guide/notebooks/Chapter_13_Ising-Model-in-Warp.ipynb` - Main notebook file for Chapter 13
- `Accelerated_Python_User_Guide/notebooks/images/chapter-13/` - Directory for chapter-specific images and diagrams
- `Accelerated_Python_User_Guide/notebooks/output/` - Directory for generated GIF animations
- `Accelerated_Python_User_Guide/notebooks/images/ising_animation.gif` - Example animation from source notebooks
- `Accelerated_Python_User_Guide/notebooks/images/checkerboard_figure.png` - Checkerboard pattern diagram
- `Accelerated_Python_User_Guide/notebooks/sources/ising-checkerboard-debug.py` - Debug mode example script

### Notes

- This notebook should maintain consistency with Chapter 12's style and structure
- Exercise solutions should be implemented with progressive reveal mechanisms
- Performance measurements should be conducted on consistent hardware for fair comparisons
- All code cells should include type hints and be compatible with Warp's requirements
- Use the existing Ising model notebooks as reference for implementation details

## Tasks

- [ ] 1.0 Set up notebook structure and introductory content
  - [x] 1.1 Create Chapter_13_Ising-Model-in-Warp.ipynb with Apache 2.0 license header
  - [x] 1.2 Write Chapter 13 title and overview section connecting to Chapter 12
  - [x] 1.3 Create Setup section with pip install commands for warp-lang, matplotlib, ipympl, PIL
  - [x] 1.4 Add organized import statements and wp.init() with GPU verification
  - [x] 1.5 Write introduction explaining how this chapter applies Chapter 12 concepts
  - [x] 1.6 Create images/chapter-13/ directory and copy relevant images from source notebooks
  - [x] 1.7 Add section separators ("---") following Chapter 12 style

- [ ] 2.0 Implement serial Python version and GPU implementations
  - [x] 2.1 Add Background section explaining 2D Ising model with accessible physics
  - [x] 2.2 Include mathematical formulation and Metropolis-Hastings algorithm explanation
  - [x] 2.3 Implement basic serial Python version with initialize_lattice and monte_carlo_step functions
  - [ ] 2.4 Create "The most naive version" section introducing parallel approach challenges
  - [ ] 2.5 Implement naive GPU version with race conditions (single array update)
  - [ ] 2.6 Add explanation of why naive approach fails with clear diagrams
  - [ ] 2.7 Implement two-array solution and explain why it's still incorrect
  - [ ] 2.8 Implement checkerboard algorithm with detailed neighbor indexing logic
  - [ ] 2.9 Add comprehensive comments explaining GPU-specific considerations

- [ ] 3.0 Create visualizations and performance analysis
  - [ ] 3.1 Implement lattice visualization using matplotlib with viridis colormap
  - [ ] 3.2 Create GIF animation generation code for lattice evolution
  - [ ] 3.3 Implement magnetization calculation kernel using atomic operations
  - [ ] 3.4 Create magnetization vs. time plots with NVIDIA green color scheme
  - [ ] 3.5 Add checkerboard pattern visualization showing black/white decomposition
  - [ ] 3.6 Implement performance timing for serial vs. GPU implementations
  - [ ] 3.7 Create performance comparison charts with different lattice sizes
  - [ ] 3.8 Add memory usage analysis and GPU utilization discussion

- [ ] 4.0 Design and implement interactive exercises
  - [ ] 4.1 Identify 3-4 key exercise points (e.g., complete update_lattice kernel, fix array indexing)
  - [ ] 4.2 Create exercise cells with clear TODO markers and partial code
  - [ ] 4.3 Implement HTML/CSS solution reveal mechanism with hidden div elements
  - [ ] 4.4 Write detailed exercise instructions with expected learning outcomes
  - [ ] 4.5 Add hints for each exercise without giving away the solution
  - [ ] 4.6 Create verification cells to test exercise solutions
  - [ ] 4.7 Ensure exercises reinforce specific Warp concepts from Chapter 12

- [ ] 5.0 Add validation, debugging guidance, and conclusion
  - [ ] 5.1 Implement analytical Onsager solution comparison for T < T_crit
  - [ ] 5.2 Create temperature sweep analysis with error bars
  - [ ] 5.3 Add section on using Warp's debug mode with practical example
  - [ ] 5.4 Include common error messages and troubleshooting strategies
  - [ ] 5.5 Write comprehensive Summary section highlighting key lessons
  - [ ] 5.6 Add References section with paper citations and Warp documentation links
  - [ ] 5.7 Review entire notebook for consistency with Chapter 12 style
  - [ ] 5.8 Verify all code runs successfully and produces expected outputs
  - [ ] 5.9 Add final notes on applying these patterns to other parallel algorithms 