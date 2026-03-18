import os

def add_inner_title(ax, title, loc, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=True, **kwargs)
    ax.add_artist(at)
    return at


def position2grid(positions, box_size, grid_size=256, sample_percent=100):
    import numpy as np

    # === sampling ===
    n_sample = int(positions.shape[0] * sample_percent / 100)
    if n_sample < positions.shape[0]:
        idx = np.random.choice(positions.shape[0], n_sample, replace=False)
        positions = positions[idx]

    cell_size = box_size / grid_size
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # === compute indices ===
    scaled = positions / cell_size
    indices = np.floor(scaled).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)

    weights = 1.0 - (scaled - indices)
    weights_indexplus = scaled - indices

    x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
    wx, wy, wz = weights[:, 0], weights[:, 1], weights[:, 2]
    wxp, wyp, wzp = weights_indexplus[:, 0], weights_indexplus[:, 1], weights_indexplus[:, 2]

    # === wrap indices for CIC assignment ===
    x_plus = (x + 1) % grid_size
    y_plus = (y + 1) % grid_size
    z_plus = (z + 1) % grid_size

    # === scatter contributions ===
    np.add.at(grid, (x, y, z), wx * wy * wz)
    np.add.at(grid, (x_plus, y, z), wxp * wy * wz)
    np.add.at(grid, (x, y_plus, z), wx * wyp * wz)
    np.add.at(grid, (x, y, z_plus), wx * wy * wzp)
    np.add.at(grid, (x_plus, y_plus, z), wxp * wyp * wz)
    np.add.at(grid, (x_plus, y, z_plus), wxp * wy * wzp)
    np.add.at(grid, (x, y_plus, z_plus), wx * wyp * wzp)
    np.add.at(grid, (x_plus, y_plus, z_plus), wxp * wyp * wzp)

    return grid, cell_size

def plot_displacement_magnitude_projection(grid_pos, displacement, boxsize, vrange=None,
                                           projection_axis='z', slice_range=(0, 100), 
                                           grid_size=128, show_3d_view=True,title=None,
                                           save_path=None, return_fig=True):
    """
    Plot the 2D projection of the displacement field magnitude.

    Parameters:
        grid_pos (np.ndarray): Grid positions, shape (N, 3)
        displacement (np.ndarray): Displacement vectors, shape (N, 3)
        boxsize (float): Box size (Mpc/h)
        projection_axis (str): Axis to project along: 'x', 'y', or 'z'
        slice_range (tuple): Range along projection axis to include in the slice (min, max)
        grid_size (int): Grid size for 3D binning
        show_3d_view (bool): Whether to include a small 3D plot for illustration
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    assert projection_axis in ['x', 'y', 'z'], "projection_axis must be 'x', 'y', or 'z'"

    axis_index = {'x': 0, 'y': 1, 'z': 2}[projection_axis]

    # Compute displacement magnitude
    displacement = (displacement + boxsize/2) % boxsize - boxsize/2
    disp_mag = np.linalg.norm(displacement, axis=1)

    # Filter slice
    min_val, max_val = slice_range
    within_slice = (grid_pos[:, axis_index] >= min_val) & (grid_pos[:, axis_index] <= max_val)
    grid_pos_slice = grid_pos[within_slice]
    disp_mag_slice = disp_mag[within_slice]

    # Project
    axes = [0, 1, 2]
    axes.remove(axis_index)
    xi = grid_pos_slice[:, axes[0]]
    yi = grid_pos_slice[:, axes[1]]

    # Bin into 2D histogram
    H, xedges, yedges = np.histogram2d(xi, yi, bins=grid_size, weights=disp_mag_slice)
    N, _, _ = np.histogram2d(xi, yi, bins=grid_size)
    H_avg = np.divide(H, N, out=np.zeros_like(H), where=N != 0)

    # Plot 2D image
    fig, ax = plt.subplots(figsize=(4,3))
    extent = [0, boxsize, 0, boxsize]
    if vrange:
        im = ax.imshow(H_avg.T, origin='lower', extent=extent, cmap='hot', vmin=vrange[0], vmax=vrange[1])
    else:
        im = ax.imshow(H_avg.T, origin='lower', extent=extent, cmap='hot')
    ax.set_xlabel(f"{['x','y','z'][axes[0]]} (Mpc/h)")
    ax.set_ylabel(f"{['x','y','z'][axes[1]]} (Mpc/h)")
    if title:
        ax.set_title(f"{title} (slice {projection_axis} axis ∈ [{min_val}, {max_val}])")
    else:
        ax.set_title(f"Displacement Magnitude Projection (slice {projection_axis} axis ∈ [{min_val}, {max_val}])")
    plt.colorbar(im, ax=ax, label='|displacement| [Mpc/h]')
    plt.tight_layout()

    # Optional 3D schematic
    if show_3d_view:
        fig3d = plt.figure(figsize=(4, 4))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_xlim(0, boxsize)
        ax3d.set_ylim(0, boxsize)
        ax3d.set_zlim(0, boxsize)
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')

        box = np.array([[0, 0, 0], [boxsize, boxsize, boxsize]])
        for i in np.linspace(min_val, max_val, 2):
            slice_marker = np.zeros((5, 3))
            slice_marker[:, axis_index] = i
            slice_marker[:, axes[0]] = [0, boxsize, boxsize, 0, 0]
            slice_marker[:, axes[1]] = [0, 0, boxsize, boxsize, 0]
            ax3d.plot(slice_marker[:,0], slice_marker[:,1], slice_marker[:,2], color='r')

        ax3d.set_title('Slice Range')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return None

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
        return None


def merge_figures(fig_list, layout=(1, 3), dpi=100, titles=None, annotations=None):

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    import numpy as np
    nrow, ncol = layout
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), dpi=dpi)
    axs = np.array(axs).reshape(-1)

    for i, f in enumerate(fig_list):
        
        from io import BytesIO
        buf = BytesIO()
        f.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = plt.imread(buf)

        # Show image
        axs[i].imshow(img)
        axs[i].axis('off')

        if titles:
            axs[i].set_title(titles[i], fontsize=12)

        if annotations:
            axs[i].text(0.5, -0.12, annotations[i],
                        fontsize=10, color='black',
                        transform=axs[i].transAxes,
                        ha='center', va='top')

    plt.tight_layout()
    return fig


def compute_projection(pos, box_size, grid_size=256, sample_percent=100,
                       projection_plane="XY", slice_range=None):

    from NeuralNbody.visualization.particle_position_plotter import position2grid

    # 3D density grid
    par_num_per_grid, cell_size = position2grid(
        pos, box_size, grid_size=grid_size, sample_percent=sample_percent)


    axis_index = {"XY": 2, "XZ": 1, "YZ": 0}[projection_plane.upper()]

    if slice_range:
        low, high = slice_range
        low_idx  = int(low / box_size * grid_size)
        high_idx = int(high / box_size * grid_size)
        slicer = [slice(None)] * 3
        slicer[axis_index] = slice(low_idx, high_idx)
        par_num_per_grid = par_num_per_grid[tuple(slicer)]

    projection = par_num_per_grid.sum(axis=axis_index)

    return projection, cell_size
    
def position_2d_plotter(positions=None, box_size=100.0, grid_size=256, projection_plane="XY",
                        sample_percent=100, redshift=None, save_path=None, vrange=None,
                        slice_range=None, show_3d_view=False,
                        positions_list=None, titles=None, ncols=3, return_fig=True):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    if positions_list is None:
        positions_list = [positions]

    n = len(positions_list)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols + 0.5, 5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.03], wspace=0.2)

    axis_index = {'XY': 2, 'XZ': 1, 'YZ': 0}[projection_plane]
    im = None

    for i, pos in enumerate(positions_list):
        row, col = i // ncols, i % ncols
        ax = fig.add_subplot(gs[row, col])
        projection, cell_size = compute_projection(pos, box_size, grid_size=grid_size,
            sample_percent=sample_percent,
            projection_plane=projection_plane,
            slice_range=slice_range)

        if vrange:
            im = ax.imshow(projection, cmap='hot', origin="lower", interpolation='bicubic', vmin=vrange[0], vmax=vrange[1])
        else:
            vm_range = pos.shape[0] / grid_size**2
            im = ax.imshow(projection, cmap='hot', origin="lower", interpolation='bicubic', vmin=vm_range * 2, vmax=vm_range * 25)


        ticks = np.linspace(0, grid_size, 5)
        # tick_labels = np.round(ticks * cell_size, 1)
        tick_labels = np.linspace(0, box_size, 5)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(['{:.0f}'.format(t) for t in tick_labels])
        ax.set_yticklabels(['{:.0f}'.format(t) for t in tick_labels])
        ax.set_xlabel('Mpc/h')
        ax.set_ylabel('Mpc/h')

        title = ''
        if titles and i < len(titles):
            title += titles[i]
        if slice_range:
            title += f'  Slice {projection_plane} ∈ [{slice_range[0]}, {slice_range[1]}]'
        ax.set_title(title)

        if redshift is not None:
            if isinstance(redshift, list) or isinstance(redshift, np.ndarray):
                if i < len(redshift):
                    add_inner_title(ax, f'z={redshift[i]}', loc='upper right')
            else:
                add_inner_title(ax, f'z={redshift}', loc='upper right')

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(fig.add_subplot(gs[j // ncols, j % ncols]))

    cbar_ax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cbar_ax, label=r'particle per grid')

    if show_3d_view and slice_range:
        fig3d = plt.figure(figsize=(5, 5))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_xlim(0, box_size)
        ax3d.set_ylim(0, box_size)
        ax3d.set_zlim(0, box_size)
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')

        min_val, max_val = slice_range
        slice_marker = np.zeros((5, 3))
        slice_marker[:, axis_index] = [min_val] * 5
        other_axes = [0, 1, 2]
        other_axes.remove(axis_index)
        slice_marker[:, other_axes[0]] = [0, box_size, box_size, 0, 0]
        slice_marker[:, other_axes[1]] = [0, 0, box_size, box_size, 0]
        ax3d.plot(slice_marker[:, 0], slice_marker[:, 1], slice_marker[:, 2], color='r')
        slice_marker[:, axis_index] = [max_val] * 5
        ax3d.plot(slice_marker[:, 0], slice_marker[:, 1], slice_marker[:, 2], color='r')
        ax3d.set_title('Slice Range')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return None

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
        return None


def position_2d_animation(positions_series,axis="XY"):
    print('not finished')



def position_3d_plotter(positions,box_size, grid_size=256,sample_percent=100, redshift=None):
    from mayavi import mlab
    density,_ = position2grid(positions,box_size, grid_size=grid_size,sample_percent=sample_percent)

    mlab.figure(size=(800, 600))

    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(density))

    mlab.colorbar(vol, title='Density')
    mlab.show()


def position_3d_animation(positions_series, box_size, redshift_series, output_gif="density_evolution.gif",grid_size=256,sample_percent=100):
    from mayavi import mlab
    mlab.options.offscreen = True
    os.makedirs("temp_frames", exist_ok=True)

    for i in range(len(positions_series)):
        fig = mlab.figure(size=(800, 600))
        density,_ = position2grid(positions_series[i],box_size, grid_size=grid_size,sample_percent=sample_percent)
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(density))
        mlab.colorbar(vol, title='Density')

        text_z = f"Redshift = {redshift_series[i]:.2f}"
        text_box = f"Box Size: {box_size} Mpc/h"

        mlab.text(0.75, 0.85, text_z, color=(0, 0, 0))
        mlab.text(0.75, 0.80, text_box, color=(0, 0, 0))

        mlab.savefig(f"temp_frames/frame_{i:04d}.png", magnification=2)
        mlab.close(fig)

    os.system(f"convert -delay 50 -loop 0 temp_frames/frame_*.png {output_gif}")
    print(f"animation saved at: {output_gif}")
    
    for file in os.listdir("temp_frames"):
        os.remove(f"temp_frames/{file}")
    os.rmdir("temp_frames")



def plot_residual_slice(pos_true, pos_pred, box_size,
                        slice_range=None, projection_plane="XY",
                        grid_size=256, sample_percent=100,
                        cmap="seismic"):

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    proj_true, _ = compute_projection(pos_true, box_size,
                                      grid_size=grid_size,
                                      sample_percent=sample_percent,
                                      projection_plane=projection_plane,
                                      slice_range=slice_range)
    proj_pred, _ = compute_projection(pos_pred, box_size,
                                      grid_size=grid_size,
                                      sample_percent=sample_percent,
                                      projection_plane=projection_plane,
                                      slice_range=slice_range)
    
    residual = proj_pred - proj_true

    fig, ax = plt.subplots()
    norm = SymLogNorm(linthresh=1e-3, vmin=residual.min(), vmax=residual.max())
    im = ax.imshow(residual, origin="lower", cmap=cmap, norm=norm)

    ax.set_title(f"Residual Slice (Pred - True), plane={projection_plane}")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Δρ")

    return fig


def field_2d_plotter(
    fields=None,
    box_size=500.0,
    grid_size=256,
    redshift=None,
    projection_plane="XY",
    slice_range=None,
    output_type="delta",
    project_mode="mean",
    titles=None,
    ncols=3,
    vrange=None,
    save_path=None,
    return_fig=True,
    show_3d_view=False,
    cmap="hot",
    interpolation="bicubic",
):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    def _ensure_list(x):
        if x is None:
            raise ValueError("fields is empty")
        return x if isinstance(x, (list, tuple)) else [x]

    def _axis_index(plane):
        return {"XY": 2, "XZ": 1, "YZ": 0}[plane.upper()]

    def _zoom_iso(arr, tgt):
        if arr.ndim not in (2, 3):
            raise ValueError("dim should 2 or 3")
        try:
            from scipy.ndimage import zoom
            if arr.ndim == 3:
                zz = tgt / arr.shape[0]
                zy = tgt / arr.shape[1]
                zx = tgt / arr.shape[2]
                return zoom(arr, (zz, zy, zx), order=1)
            else:
                zy = tgt / arr.shape[0]
                zx = tgt / arr.shape[1]
                return zoom(arr, (zy, zx), order=1)
        except Exception:

            if arr.ndim == 3:
                zi = (np.arange(tgt) * (arr.shape[0] / tgt)).astype(int).clip(0, arr.shape[0]-1)
                yi = (np.arange(tgt) * (arr.shape[1] / tgt)).astype(int).clip(0, arr.shape[1]-1)
                xi = (np.arange(tgt) * (arr.shape[2] / tgt)).astype(int).clip(0, arr.shape[2]-1)
                return arr[zi[:, None, None], yi[None, :, None], xi[None, None, :]]
            else:
                yi = (np.arange(tgt) * (arr.shape[0] / tgt)).astype(int).clip(0, arr.shape[0]-1)
                xi = (np.arange(tgt) * (arr.shape[1] / tgt)).astype(int).clip(0, arr.shape[1]-1)
                return arr[yi[:, None], xi[None, :]]

    def _phys_to_index_range(axis_len, rng_phys, Lbox):
        if rng_phys is None:
            return (0, axis_len)
        a, b = float(rng_phys[0]), float(rng_phys[1])
        a = max(0.0, min(Lbox, a))
        b = max(0.0, min(Lbox, b))
        if b < a:
            a, b = b, a
        i0 = int(np.floor(a / Lbox * axis_len))
        i1 = int(np.ceil (b / Lbox * axis_len))
        i0 = max(0, min(axis_len, i0))
        i1 = max(0, min(axis_len, i1))
        return (i0, max(i0+1, i1))

    def _to_2d_from_3d(vol3d, plane, mode, slice_rng):
        ax = _axis_index(plane)
        if slice_rng is not None:
            i0, i1 = _phys_to_index_range(vol3d.shape[ax], slice_rng, box_size)
            sl = [slice(None)] * 3
            sl[ax] = slice(i0, i1)
            vol3d = vol3d[tuple(sl)]
        if mode == "mean":
            proj = vol3d.mean(axis=ax)
        elif mode == "sum":
            proj = vol3d.sum(axis=ax)
        else:
            raise ValueError(f"Unknown project_mode: {mode}")
        return proj

    def _auto_vrange(a2d, kind):

        q1, q99 = np.nanpercentile(a2d, [1, 99])
        if kind == "delta":
            a = max(abs(q1), abs(q99))
            a = a if np.isfinite(a) and a > 0 else 1.0
            return (-a, a)
        if not np.isfinite(q1): q1 = np.nanmin(a2d)
        if not np.isfinite(q99): q99 = np.nanmax(a2d)
        if q1 == q99:
            return (q1 - 1e-6, q99 + 1e-6)
        return (q1, q99)

    def _cbar_label(kind, mode):
        if kind == "delta":
            return r"$\delta$"
        if kind == "rho":
            return r"$\rho$" if mode == "mean" else r"$\Sigma\;(\int \rho\,\mathrm{d}\ell)$"
        return "value"

    arrays = _ensure_list(fields)
    arrays = [np.asarray(a) for a in arrays]

    arrays_rs = []
    for a in arrays:
        if a.shape[0] == a.shape[1] == a.shape[2] == grid_size:
            arrays_rs.append(a.astype(np.float32, copy=False))
        else:
            arrays_rs.append(_zoom_iso(a, grid_size).astype(np.float32, copy=False))


    n = len(arrays_rs)
    ncols = min(max(1, ncols), n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols + 0.5, 5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.03], wspace=0.2)

    axis_index = _axis_index(projection_plane)
    im = None


    for i, vol in enumerate(arrays_rs):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])

        img2d = _to_2d_from_3d(vol, projection_plane, project_mode, slice_range)


        if vrange is not None:
            vmin, vmax = vrange
        else:
            vmin, vmax = _auto_vrange(img2d, output_type)

        im = ax.imshow(
            img2d, cmap=cmap, origin="lower",
            interpolation=interpolation, vmin=vmin, vmax=vmax
        )

        ticks = np.linspace(0, grid_size, 5)
        tick_labels = np.linspace(0, box_size, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(['{:.0f}'.format(t) for t in tick_labels])
        ax.set_yticklabels(['{:.0f}'.format(t) for t in tick_labels])
        ax.set_xlabel('Mpc/h')
        ax.set_ylabel('Mpc/h')

        if redshift is not None:
            if isinstance(redshift, (list, tuple, np.ndarray)) and i < len(redshift):
                add_inner_title(ax, f'z={redshift[i]:.1f}', loc='upper right')
            elif not isinstance(redshift, (list, tuple, np.ndarray)):
                add_inner_title(ax, f'z={redshift:.1f}', loc='upper right')

        title = ''
        if titles and i < len(titles):
            title += str(titles[i])
        if slice_range is not None:
            title += ('' if title == '' else ' | ')
            title += f'Slice {projection_plane.upper()} ∈ [{slice_range[0]}, {slice_range[1]}]'
        if project_mode in ("mean", "sum"):
            title += ('' if title == '' else ' | ') + f'Proj={project_mode}'
        if output_type:
            title += ('' if title == '' else ' | ') + f'Out={output_type}'
        ax.set_title(title)

    last_i = n - 1
    for j in range(last_i + 1, nrows * ncols):
        fig.delaxes(fig.add_subplot(gs[j // ncols, j % ncols]))

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax, label=_cbar_label(output_type, project_mode))

    if show_3d_view and slice_range is not None:
        fig3d = plt.figure(figsize=(5, 5))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_xlim(0, box_size)
        ax3d.set_ylim(0, box_size)
        ax3d.set_zlim(0, box_size)
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')

        min_val, max_val = slice_range

        slice_marker = np.zeros((5, 3))
        slice_marker[:, axis_index] = [min_val] * 5
        other_axes = [0, 1, 2]
        other_axes.remove(axis_index)
        slice_marker[:, other_axes[0]] = [0, box_size, box_size, 0, 0]
        slice_marker[:, other_axes[1]] = [0, 0, box_size, box_size, 0]
        ax3d.plot(slice_marker[:, 0], slice_marker[:, 1], slice_marker[:, 2], color='r')
        slice_marker[:, axis_index] = [max_val] * 5
        ax3d.plot(slice_marker[:, 0], slice_marker[:, 1], slice_marker[:, 2], color='r')
        ax3d.set_title('Slice Range')

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return None

    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
        return None


#### figure plot
# filename = "/mnt/r/SimML/NeuralNbody/simulation/Gadget4/simulations_128_100/nbody1/snapshot_004.hdf5"

# ptype=[1]
# header   = readgadget.header(filename)
# BoxSize  = header.boxsize  #Mpc/h
# redshift = header.redshift
# positions = readgadget.read_block(filename, "POS ", ptype) # Mpc/h

# position_2d_plotter(positions,BoxSize, redshift=redshift, grid_size=256,projection_plane="XY",sample_percent=100)
# position_3d_plotter(positions,BoxSize,redshift=redshift, grid_size=128)


#### animation
# positions_series = []
# redshift_series = []
# for i in range(1,5):
#     filename = f"/mnt/r/SimML/NeuralNbody/simulation/Gadget4/simulations_128_100/nbody1/snapshot_00{i}.hdf5"
#     ptype=[1]
#     header   = readgadget.header(filename)
#     BoxSize  = header.boxsize  #Mpc/h
#     redshift = header.redshift
#     positions = readgadget.read_block(filename, "POS ", ptype) # Mpc/h
#     positions_series.append(positions)
#     redshift_series.append(redshift)

# position_3d_animation(positions_series, BoxSize,redshift_series, grid_size=256)