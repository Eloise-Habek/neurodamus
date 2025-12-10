import numpy as np
import matplotlib.pyplot as plt
import bluepysnap as bp
from sklearn.decomposition import PCA
from numpy import pi
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def load_data(sim_config_path, node_group):
    try:
        s = bp.Simulation(sim_config_path)
    except Exception:
        return None, None, None, None

    try:
        allData = s.reports["soma"]["S1nonbarrel_neurons"].get()
    except Exception:
        allData = None

    try:
        spikeData = s.spikes["S1nonbarrel_neurons"].get()
    except Exception:
        spikeData = None

    try:
        cellData = s.circuit.nodes["S1nonbarrel_neurons"].get(group=node_group)
    except Exception:
        cellData = None

    return s, allData, cellData, spikeData


def define_central_axis(somaPos):
    ## code from blue brain project - BlueRecording

    center = np.mean(somaPos, axis=0).values

    pca = PCA(n_components=3)
    pca.fit(somaPos)
    main_axis = pca.components_[0]

    elevation = np.arctan2(main_axis[2], np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2))
    azimuth = np.arctan2(main_axis[1], main_axis[0])

    return center, azimuth, elevation, main_axis


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix that rotates vector a to vector b.
    Both a and b must be unit vectors.
    Uses Rodrigues' rotation formula. -- Chat generated
    """
    # cross and dot
    v = np.cross(a, b)
    c = np.dot(a, b)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        # a and b are parallel (could be same or opposite)
        if c > 0:  # same direction
            return np.eye(3)
        else:  # opposite direction: rotate 180 degrees around any orthogonal axis
            # find an orthogonal vector
            orth = np.array([1, 0, 0], dtype=float)
            if abs(a[0]) > 0.9:
                orth = np.array([0, 1, 0], dtype=float)
            u = orth - np.dot(orth, a) * a
            u = u / np.linalg.norm(u)
            # rotation by pi around u: R = I - 2 uu^T
            return np.eye(3) - 2.0 * np.outer(u, u)

    k = v / v_norm
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def get_somaPos(simulationData):

    c = simulationData.circuit
    n = c.nodes

    return n["S1nonbarrel_neurons"].get(properties=["x", "y", "z"])


def get_cell_to_rank(cellData, simulationData):

    somaPosition = get_somaPos(simulationData)
    center, azimuth, elevation, pca1 = define_central_axis(somaPosition)
    pca1 = pca1 / np.linalg.norm(pca1)

    z_axis = np.array([0.0, 0.0, 1.0])

    R = rotation_matrix_from_vectors(pca1, z_axis)  # rotates pca1 -> z
    X_rot = somaPosition @ R.T
    X_rot.columns = ["x", "y", "z"]
    cellData.update(X_rot[["x", "y", "z"]])
    cellData.sort_values(by="z", ascending=False, inplace=True)
    cellData["rank"] = cellData["z"].rank(method="first", ascending=True).astype(int)
    cell_to_rank = cellData["rank"]

    return cell_to_rank


def get_estim_data(simulationData):
    try:
        eStim = simulationData.config["inputs"]["estim"]
        x = eStim["rise_time"]
        y = eStim["decay_time"]
        z = eStim["mean_percent"]
        f = eStim["dt"]
        delay = eStim["delay"]
        duration = eStim["duration"]
        # in ms the sampling frequency - 10_000 Hz

        somaPosition = get_somaPos(simulationData)
        center, azimuth, elevation, pca1 = define_central_axis(somaPosition)

        amp = x / pca1[0]
        print(eStim)
    except Exception as e:
        print(e)
        return None

    return x, y, z, f, delay, duration, amp


def define_psth_plot(simulation_time, title, cell_to_rank):
    fig, (ax_raster, ax_stimulation) = plt.subplots(
        2, 1, figsize=(16, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Neurons")
    ax_raster.set_title(f"Raster plot - {title}")
    ax_raster.grid(False)
    ax_raster.set_ylim(0, cell_to_rank.shape[0])

    ax_hist = ax_raster.twinx()

    ax_hist.set_ylabel("Populaion Firing rate (Hz)")
    ax_hist.set_ylim(0, 1)

    ax_stimulation.set_xlabel("Time (ms)")
    ax_stimulation.set_ylabel("Amplitude (V/m)")
    ax_stimulation.set_title("Electric field stimulation")
    ax_stimulation.set_ylim(-2.2, 2.2)

    ax_stimulation.grid()

    for ax in (ax_raster, ax_stimulation):
        ax.set_xlim(0, simulation_time)

    ax_raster.tick_params(axis="x", labelbottom=True, bottom=True)
    return fig, (ax_raster, ax_stimulation, ax_hist)


def get_firingrate_hist(
    simulationData,
    spikeData,
    bin_width,
    n_neurons,
    delay=None,
    time_stop=None,
    center=True,
):
    # Population firing rate
    stimulation_info = get_estim_data(simulationData)

    if delay is None:
        if stimulation_info is None:
            delay = 0
        else:
            x, y, z, f, delay, duration, amp = stimulation_info

    if time_stop is None:
        time_stop = simulationData.time_stop
    bins = np.arange(delay, time_stop + bin_width, bin_width)  # in ms

    all_spike = spikeData.index.tolist()
    hist_counts, _ = np.histogram(all_spike, bins=bins)
    bin_seconds = bin_width * 1e-3
    firing_rate = hist_counts / bin_seconds / n_neurons

    if center:
        bins = bins[1:] - bin_width / 2

    return firing_rate, bins


def define_stimulation_function(simulationData, dt):
    stimulation_info = get_estim_data(simulationData)
    simulation_time = simulationData.time_stop
    time = np.arange(0, simulation_time, dt)

    if stimulation_info == None:
        stimulation_plot, f, amp = np.zeros((time.shape[0],)), 0, 0

    else:
        x, y, z, f, delay, duration, amp = stimulation_info

        if simulation_time < delay:
            stimulation_plot, f, amp = np.zeros((time.shape[0],)), 0, 0
            return stimulation_plot, f, amp

        if simulation_time - delay < duration:
            duration = simulation_time - delay

        if f == 0:
            print("DC stimulation")
            duration_time = np.arange(0, duration, dt)
            stimulation_plot = np.ones((duration_time.shape[0],)) * amp
        else:
            print("AC stimulation")
            stimulation_plot = amp * np.sin(
                2 * pi * f * 10e-4 * np.arange(delay, delay + duration, dt)
            )

        if delay > 0:
            delay_time = np.arange(0, delay, dt)
            delay_plot = np.zeros((delay_time.shape[0],))

            stimulation_plot = np.concatenate((delay_plot, stimulation_plot))

        if (delay + duration) < simulation_time:
            delay_time = np.arange(0, simulation_time - delay - duration, dt)
            delay_plot = np.zeros((delay_time.shape[0],))
            stimulation_plot = np.concatenate((stimulation_plot, delay_plot))

    return stimulation_plot, f, amp


def psth_plot(
    spikeData,
    title,
    cell_to_rank,
    simulationData,
    dt=0.1,
    bin_width=1,
    save_histogram=None,
    smoothed=False,
):

    simulation_time = simulationData.time_stop

    fig, (ax_raster, ax_stimulation, ax_hist) = define_psth_plot(
        simulation_time, title, cell_to_rank
    )

    cell_to_rank = pd.Series(
        range(len(cell_to_rank) - 1, -1, -1), index=cell_to_rank.index
    )
    spikeData_ranked = spikeData.map(cell_to_rank)
    spikes = spikeData_ranked.reset_index().to_numpy()
    cell_ids_rank = spikes[:, 1]
    spike = spikes[:, 0]

    # plot raster plot
    time = np.arange(0, simulation_time, dt)
    ax_raster.scatter(spike, cell_ids_rank, s=0.5, c="black")
    # create histogram of spike train
    firing_rate, bins = get_firingrate_hist(
        simulationData,
        spikeData_ranked,
        bin_width,
        cell_to_rank.shape[0],
        delay=0,
        time_stop=simulation_time,
    )

    # create stimulation plot
    stimulation_plot, f, amp = define_stimulation_function(simulationData, dt)
    # plot stimulation function
    amp = round(amp, 0)
    ax_hist.step(bins, firing_rate, where="mid", label=f"{f}Hz{amp}Vm")

    if smoothed:
        sigma_ms = 10.0  # adjusted for frequency resolution up to 15Hz
        sigma_samples = sigma_ms / bin_width
        smoothed_fr = gaussian_filter1d(firing_rate, sigma=sigma_samples)
        ax_hist.plot(
            bins,
            smoothed_fr,
            color="C1",
            lw=1,
            label=f"Gaussian smoothed, σ={sigma_ms} ms",
        )

    ax_stimulation.plot(time, stimulation_plot)

    plt.tight_layout()
    if save_histogram is not None:
        plt.savefig(save_histogram)
    return fig, (ax_raster, ax_stimulation, ax_hist)
