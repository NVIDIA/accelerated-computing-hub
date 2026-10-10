# Online GPU teaching checks

Both completed Article 2 teaching examples passed on the workstation: SO-101 and reBot each ran one environment for 2,000 control frames (40 simulated seconds), with 20 physics substeps per frame. Both cubes were grasped, lifted, carried and released inside the receiving box, remained contained and settled, and the gripper withdrew.

These examples use the online CPU controller and GPU physics. They validate the runnable teaching solutions; their execution logs are not performance measurements and are excluded from the batch benchmark tables and ratios.

The exact commands, start/end records, unmodified task reports and logs are retained here. The solution and box-task source identities are recorded in `manifest.json` and match the benchmark package's `../source.tar.gz`. These checks establish the recorded workstation runs only; they do not establish validation on Colab, Brev or another GPU.
