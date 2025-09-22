import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import glob
import os


def run(filename):
    for file in glob.glob('/tmp/heat_*'):
        os.remove(file)

    if os.path.exists('/tmp/a.out'):
        os.remove('/tmp/a.out')

    # Execute commands above
    result = subprocess.run(
        ['nvcc', '-x', 'cu', '-arch=native', '--extended-lambda', '-o', '/tmp/a.out', filename],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    subprocess.run(['/tmp/a.out'])

    img = None
    fig = plt.figure(figsize=(8, 2))
    fig.tight_layout()

    def drawframe(i):
        with open(f"/tmp/heat_{i}.bin", 'rb') as f:
            height, width = np.fromfile(f, dtype=np.int32, count=2)
            data = np.fromfile(f, dtype=np.float32, count=height * width)
            data = data.reshape((height, width))

        nonlocal img
        if img is None:
            img = plt.imshow(data, cmap='hot', interpolation='none', vmin=10)
        else:
            img.set_data(data)
        return img,

    ani = animation.FuncAnimation(
        fig, drawframe, frames=100, interval=20, blit=True)
    plt.close(fig)  # Suppress the figure display
    return HTML(ani.to_html5_video())
