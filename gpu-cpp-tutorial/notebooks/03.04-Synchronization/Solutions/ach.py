import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import glob
import os


def run(filename, only_show=False):
    if not only_show:
        # Clean up any previous run files
        for file in glob.glob('/tmp/heat_*'):
            os.remove(file)
        for file in glob.glob('/tmp/hist_*'):
            os.remove(file)

        if os.path.exists('/tmp/a.out'):
            os.remove('/tmp/a.out')

        # Compile and run
        result = subprocess.run(
            ['nvcc', '-x', 'cu', '-arch=native', '--extended-lambda',
                '-o', '/tmp/a.out', filename]
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
        subprocess.run(['/tmp/a.out'])

    # Create figure with two subplots:
    #   ax1 -> for temperature map
    #   ax2 -> for histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, bottom=0.15)

    img = None
    bar_container = None

    def drawframe(i):
        # Read the temperature data
        with open(f"/tmp/heat_{i}.bin", 'rb') as f:
            height, width = np.fromfile(f, dtype=np.int32, count=2)
            data = np.fromfile(f, dtype=np.float32, count=height * width)
            data = data.reshape((height, width))

        # Read the histogram data
        with open(f"/tmp/hist_{i}.bin", 'rb') as f:
            bins, = np.fromfile(f, dtype=np.int32, count=1)
            hist = np.fromfile(f, dtype=np.int32, count=bins)

        nonlocal img, bar_container

        # If first frame, create the image and the bar plot
        if img is None:
            # (a) Show heat map
            img = ax1.imshow(data, cmap='hot', interpolation='none', vmin=10)
            ax1.set_title("Temperature Map")

            # (b) Histogram
            #     We assume the temperature goes from 0 up to 100
            #     so the bin size is 100 / bins, and edges are [0, bin_size, ..., 100].
            bin_edges = np.linspace(0, 100, bins + 1)

            # Plot the histogram bars at x = 0..bins-1
            bar_container = ax2.bar(
                range(bins), hist, width=1.0, align='center')
            ax2.set_title("Histogram")

            # Set up x-axis ticks: center each bar at an integer (0..bins-1)
            # Then label them with the corresponding temperature ranges.
            tick_positions = np.arange(bins)
            tick_labels = [
                f"{bin_edges[j]:.0f} - {bin_edges[j+1]:.0f}"
                for j in range(bins)
            ]

            # Assign them
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)  # , rotation=45, ha='right')

            # Make sure bars/labels are not cut off
            ax2.set_xlim(-0.5, bins - 0.5)
            ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax2.set_xlabel("Temperature Range")
            ax2.set_ylabel("Number of Cells")

        else:
            # Update the heat map
            img.set_data(data)

            # Update the histogram bar heights
            for rect, h in zip(bar_container, hist):
                rect.set_height(h)

        # Return all artists that need updating
        return [img, *bar_container]

    ani = animation.FuncAnimation(
        fig, drawframe, frames=100, interval=50, blit=False
    )
    return HTML(ani.to_html5_video())
