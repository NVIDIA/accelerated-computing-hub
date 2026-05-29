# Easy GPU Acceleration Wins with RAPIDS

This section shows how to take existing CPU Python workflows and try GPU
acceleration without rewriting the application. We will stay in the terminal and
run scripts directly. Notebook usage is covered separately.

You will try two accelerators:

- `cudf.pandas` for pandas-style dataframe work.
- `cuml.accel` for scikit-learn-style machine learning work.

The goal is not to tune every line of code. The goal is to quickly answer:

- Does my existing script run with the accelerator?
- Which parts ran on the GPU?
- Did the runtime improve enough to investigate further?

## Setup checkup

```bash
nvidia-smi
```

Make sure you have an active environment that has `cudf`, `cuml`, `pandas` and 
`sklearn`. 


Confirm that these packages are available in the Python environment you are
using:

```bash
python -c "import cudf, cuml, pandas, sklearn; print('environment is ready')"
```

**TODO: DECIDE IF WE SHOULD DOWNLOAD THIS AS PART OF SETUP**

Download the dataset used by the pandas workflow:

```bash
python scripts/data-setup.py --nyc-parking
```

## pandas Workflow: Baseline CPU Run

Imagine you already have pandas code in a script and want to know whether it can
benefit from a GPU.

First inspect the existing pandas workflow. For example, in `pandas-workflow.py`
we read the NYC parking violations data and runs common dataframe
operations such as `value_counts`, `groupby`, `agg`, datetime extraction,
sorting, and `count`.

Run it normally, to see the cpu baseline:

```bash
python scripts/pandas-workflow.py
```

## Accelerate pandas with `cudf.pandas`

If `cudf` is installed, try the same script with zero code changes:

```bash
python -m cudf.pandas scripts/pandas-workflow.py
```

That is the main workflow: replace `python` with `python -m cudf.pandas`.

Your script still imports `pandas`, but `cudf.pandas` intercepts pandas imports
and uses cuDF on the GPU where possible. Operations that are not supported on
the GPU fall back to pandas on the CPU.

For a quick comparison, run each path a few times:

```bash
for i in 1 2 3; do python scripts/pandas-workflow.py; done
for i in 1 2 3; do python -m cudf.pandas scripts/pandas-workflow.py; done
```

## Understanding cudf.pandas Performance

**Exercise:** What do you notice? When comparing the runs, keep these points in
mind:

- The first GPU run may include startup and GPU context initialization overhead.
- The script reads the parquet file before the timed operations, so the printed
  timings focus on dataframe operations rather than download or file I/O.
- Larger dataframe operations usually show the clearest speedups.
- Very small operations can be dominated by overhead and may not be faster.
- Unsupported pandas operations can fall back to CPU, which is correct but may
  reduce the speedup.

We can, have more insight on what's happening using the the built-in 
`cudf.pandas` profilers.

We have the line profiler that shows the source code and how much time each line
spent executing on the GPU and CPU.

```bash
python -m cudf.pandas --line-profile scripts/pandas_workflow.py
```

and if we use `--profile`, it generates a report showing which operations used 
the GPU and which used the CPU.

```bash
python -m cudf.pandas --profile scripts/pandas_workflow.py
```

Look for:

- Operations that ran on the GPU.
- Operations that fell back to CPU.
- Lines or functions that dominate total runtime.

If a row-wise or Python-object-heavy operation falls back to CPU, that is often
the next place to simplify the workflow.


## scikit-learn Workflow: Baseline CPU Run

Similarly, we have `cuml.accel` for `scikit-learn`, `UMAP`, and `HDBSCAN`

The `sklearn-workflow.py` script generates a synthetic classification dataset,
trains a `RandomForestClassifier`, predicts on a test split, and prints accuracy.

Run the CPU baseline:

```bash
python scripts/sklearn-workflow.py
```

## Accelerate scikit-learn with `cuml.accel`

If `cuml` is installed, try the same script with zero code changes:

```bash
python -m cuml.accel scripts/sklearn-workflow.py
```

Again, the script still imports from `sklearn`. The accelerator dispatches
supported estimators and methods to cuML on the GPU and falls back to CPU where
needed.

Run both paths a few times:

```bash
for i in 1 2 3; do python scripts/sklearn-workflow.py; done
for i in 1 2 3; do python -m cuml.accel scripts/sklearn-workflow.py; done
```

The accuracy does not need to be bit-for-bit identical between CPU and GPU
implementations. Compare model quality, not internal fitted attributes.

## Understanding scikit-learn Performance

Machine learning acceleration depends on the estimator, hyperparameters, data
size, and data types.

`cuml.accel` also counts with profilers to get better understanding on what's being accelerated:

Use the function profiler for a compact report:

```bash
python -m cuml.accel --profile scripts/sklearn-workflow.py
```

Use the line profiler when you want per-line detail:

```bash
python -m cuml.accel --line-profile scripts/sklearn-workflow.py
```

Do not use profiler runs as benchmark numbers. Profilers add overhead. Use them
to understand dispatch behavior, then benchmark with the plain accelerated
command.


## Watch the GPU with Jupyterlab NVDashboard

If you work on Jupyter notebooks, the JupyterLab NVDashboard extension is a great 
tool to watch some GPU metrics like memory and utilization. 

**TODO: REVISIT WHERE TO DO THESE INSTALL. **

Install the dashboard into the Python environment used by the Brev Jupyter
server:

```bash
python -m pip install jupyterlab_nvdashboard
sudo systemctl restart jupyter
```

### Example

Let's copy the script code in the notebook so we have everything on the same 
place. Let's run the code and monitor the dashboard.

You should see GPU memory and utilization change while the accelerated sections
run. 

If we need to accelerate code or profile in the notebook, we can using jupyter
magics. 


## Takeaways

- `cudf.pandas` is the fastest way to try GPU acceleration on existing pandas code.
- `cuml.accel` is the fastest way to try GPU acceleration on supported scikit-learn,
UMAP, and HDBSCAN workflows.
- Built-in accelerator profilers explain GPU execution and CPU fallback.


## References

- [cuDF pandas accelerator usage](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/usage/)
- [cuML zero code change acceleration](https://docs.rapids.ai/api/cuml/stable/cuml-accel/)
- [cuML accelerator logging and profiling](https://docs.rapids.ai/api/cuml/stable/cuml-accel/logging-and-profiling/)
- [cuML accelerator limitations](https://docs.rapids.ai/api/cuml/stable/cuml-accel/limitations/)
- [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard)
