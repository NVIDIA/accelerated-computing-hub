import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('dark_background') # Uncomment for dark background

def visualize(name = 'output'):
    f = open(name, 'rb')
    grid = np.fromfile(f, dtype=np.uint64, count=2, offset=0)

    nx = grid[0]
    ny = grid[1]

    times = np.fromfile(f, dtype=np.float64, count=1, offset=0)
    time = times[0]

    values = np.fromfile(f, dtype=np.float64, offset=0)
    assert len(values) == nx * ny, f'{len(values)} != {nx * ny}'
    # pcolormesh plots nrows in the y axis, and ncols in the x axis
    # the number of rows is "ny", and the number of columns is "nx"
    # and we need to load the file in "Fortran" format (col-major)
    values = values.reshape((nx, ny), order='F')
    

    print(f'Plotting grid {nx}x{ny}, t = {time}')
    print(values.shape)

    plt.title(f'Temperature at t = {time:.3f} [s]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pcolormesh(values, cmap=plt.cm.jet, vmin=0.0, vmax=values.max())
    plt.colorbar()
    plt.savefig('output.png', transparent=True, bbox_inches='tight', dpi=300)