# Product Requirements Document: Chapter 13 - Ising Model in Warp

## Introduction/Overview

This PRD outlines the requirements for creating Chapter 13 of the Accelerated Python User Guide, which will demonstrate parallel algorithm design patterns using the 2D Ising model as a case study. This chapter serves as a natural continuation of Chapter 12 (Introduction to NVIDIA Warp), applying the fundamental concepts learned to solve a real-world computational physics problem while teaching important GPU programming patterns.

## Goals

1. **Teach parallel algorithm design patterns** through a practical physics simulation example
2. **Reinforce all core Warp concepts** from Chapter 12 (kernels, arrays, memory management, etc.)
3. **Demonstrate common GPU programming pitfalls** and their solutions through incremental development
4. **Provide performance comparisons** between serial Python and various GPU implementations
5. **Maintain consistency** with Chapter 12's documentation style and pedagogical approach

## User Stories

1. **As a physics/computational science student**, I want to understand how to implement the Ising model on GPU so that I can apply similar techniques to my own simulations.
2. **As an advanced user**, I want to see complex Warp examples so that I can learn best practices for parallel algorithm design.
3. **As someone who completed Chapter 12**, I want to apply my newly learned Warp skills to a real problem so that I can solidify my understanding.
4. **As a developer**, I want to understand race conditions and synchronization issues so that I can avoid common GPU programming mistakes.
5. **As a learner**, I want interactive exercises with progressive solution reveals so that I can test my understanding before seeing the answer.

## Functional Requirements

1. **The notebook must begin with a clear connection to Chapter 12**, referencing concepts already learned and explaining how they'll be applied.

2. **The notebook must include a comprehensive Setup section** similar to Chapter 12, including:
   - Required package installations (warp-lang, matplotlib, ipympl, PIL)
   - Import statements with clear organization
   - Verification of GPU availability

3. **The notebook must provide an accessible physics background** that:
   - Explains the 2D Ising model with minimal assumed physics knowledge
   - Focuses on programming concepts over deep physics theory
   - Includes the Metropolis-Hastings algorithm explanation
   - Uses clear mathematical notation consistent with Chapter 12

4. **The notebook must follow an incremental development approach**:
   - Start with a working serial Python implementation
   - Present the naive parallel approach (with race conditions)
   - Introduce the two-array solution (explaining why it's still incorrect)
   - Present the checkerboard algorithm as the correct solution
   - Include exercises at key points with solutions revealed progressively

5. **The notebook must include enhanced visualizations**:
   - Animated GIFs of lattice evolution
   - Magnetization vs. time plots
   - Clear checkerboard pattern diagrams
   - Additional diagrams explaining parallel update patterns
   - Performance comparison charts

6. **The notebook must provide comprehensive performance analysis**:
   - Time measurements for serial Python implementation
   - Comparison with naive GPU approach
   - Performance of the optimized checkerboard algorithm
   - Discussion of scaling with lattice size

7. **The notebook must maintain Chapter 12's documentation style**:
   - Use "---" separators between major sections
   - Include clear section headers (Setup, Background, Implementation, etc.)
   - Provide inline explanations after code cells
   - Use code citations in the format ```startLine:endLine:filepath
   - Include a comprehensive references section

8. **The notebook must include debugging guidance**:
   - Show how to use Warp's debug mode
   - Explain common error messages
   - Demonstrate debugging strategies for parallel algorithms

9. **The notebook must reinforce Warp concepts through practical application**:
   - Kernel definition and launching
   - Array allocation and management
   - Type conversions and constraints
   - Random number generation in kernels
   - Atomic operations for reductions
   - Memory access patterns

10. **The notebook must conclude with validation**:
    - Compare results with analytical Onsager solution
    - Discuss sources of numerical differences
    - Summarize lessons learned about parallel algorithm design

## Non-Goals (Out of Scope)

1. **Will NOT include** advanced physics beyond the basic 2D Ising model
2. **Will NOT explore** alternative algorithms besides the checkerboard approach
3. **Will NOT implement** multi-GPU versions
4. **Will NOT compare** with other GPU frameworks (CuPy, Numba, etc.)
5. **Will NOT delve** into 3D Ising models or other lattice systems
6. **Will NOT include** magnetic field effects or other model extensions

## Design Considerations

1. **Code Structure**:
   - Each implementation should be self-contained and runnable
   - Variable names should be consistent across implementations
   - Comments should explain GPU-specific considerations

2. **Exercise Design**:
   - Exercises should have clear TODO markers
   - Solutions should be hidden initially (using HTML/CSS tricks)
   - Each exercise should reinforce a specific concept

3. **Visual Consistency**:
   - All plots should use consistent color schemes (matching Chapter 12 where applicable)
   - Figures should have clear captions and labels
   - Animation frame rates should be reasonable for notebook viewing

## Technical Considerations

1. **Dependencies**:
   - Must work with the same Warp version as Chapter 12
   - Should specify exact versions for reproducibility
   - Must handle missing GPU gracefully with appropriate warnings

2. **Performance**:
   - Default parameters should run reasonably fast (< 1 minute total)
   - Should provide options for larger simulations with warnings about runtime

3. **Memory Usage**:
   - Should work on GPUs with at least 4GB memory
   - Large lattice sizes should include memory requirement warnings

## Success Metrics

1. **Learning Effectiveness**: Readers successfully complete exercises and understand parallel algorithm design patterns
2. **Performance Understanding**: Readers can explain why the checkerboard algorithm is necessary and efficient
3. **Concept Reinforcement**: Readers demonstrate mastery of Warp concepts introduced in Chapter 12
4. **Practical Application**: Readers can apply similar patterns to their own parallel algorithms

## Open Questions

1. Should we include a brief section on profiling Warp kernels for the Ising model?
2. Should we add an optional advanced section on optimizations (e.g., shared memory usage)?
3. Should we include links to the original research papers in the main text or just in references?
4. Should we provide a standalone Python script version of the final implementation?
5. Should we add a troubleshooting section for common Warp errors specific to this implementation? 