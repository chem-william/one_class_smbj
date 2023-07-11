from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from numpy.random import Generator
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

sns.set(
    style="ticks",
    rc={
        "font.family": "Arial",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 5,
    },
    font_scale=2.5,
    palette=sns.color_palette("Set2"),
)

ALPHABET = "ABCDEFGHIJKLMNOPQ"
CONDUCTANCE_LABEL = r"Conductance (log$_{10}$(G/G$_0$))"
c = [
    "#007fff",
    "#ff3616",
    "#138d75",
    "#7d3c98",
    "#fbea6a",
]  # Blue, Red, Green, Purple, Yellow


@dataclass
class PreprocessInfo:
    high: float
    low: float
    plot_high: float
    plot_low: float
    bins: int
    bins_2d: Tuple[int, int]
    hist2d_length: int


def preprocess_4k_data(
    traces: np.ndarray, labels: np.ndarray, pre_info: PreprocessInfo
) -> Tuple[np.ndarray, np.ndarray, np.array]:
    return_data, fullwindow, new_labels = [], [], []
    amount_discarded = 0
    longest = 0
    for trace, label in tqdm(
        zip(traces, labels), desc="Processing traces", total=len(traces)
    ):
        trace = np.log10(trace)
        trace = trace[np.logical_not(np.isnan(trace))]
        full = trace[trace < pre_info.plot_high]
        full = full[full > pre_info.plot_low]

        trace = trace[trace < pre_info.high]
        trace = trace[trace > pre_info.low]

        if len(trace) == 0:
            amount_discarded += 1
            continue

        if label == -1:
            continue

        if len(trace) > longest:
            longest = len(trace)

        return_data.append(trace)
        fullwindow.append(full)
        new_labels.append(label)
    print(f"Longest trace: {longest}")
    print(f"Amount discarded: {amount_discarded}")

    return (
        np.array(fullwindow, dtype="object"),
        np.array(return_data, dtype="object"),
        np.array(new_labels),
    )


def get_histograms(
    traces: np.ndarray, low: float, high: float, pre_info: PreprocessInfo
) -> Tuple[np.ndarray, np.ndarray]:
    hist = []
    hist_2d = []
    for trace in tqdm(traces, desc="Generating 1D- and 2D histograms"):
        h, _ = np.histogram(trace, bins=pre_info.bins, range=(low, high))
        H, *_ = np.histogram2d(
            trace,
            np.arange(len(trace)),
            bins=pre_info.bins_2d,
            range=[[low, high], [0, pre_info.hist2d_length]],
        )
        hist.append(h)
        hist_2d.append(H.ravel())

    return np.array(hist), np.array(hist_2d)


def plot_individual_traces(traces: np.ndarray, ax, size:int, rng: Generator, **kwargs) -> None:
    rng_ints = rng.integers(low=0, high=len(traces), size=size)
    for trace in traces[rng_ints]:
        ax.plot(trace, **kwargs)


def generate_2dhistograms(traces: np.ndarray, bins_2d: int, hist2d_length: int, pre_info: PreprocessInfo) -> np.ndarray:
    hist_2d = np.zeros((bins_2d, bins_2d))
    for trace in tqdm(traces, desc="Generating 2D histograms"):
        H, *_ = np.histogram2d(
            trace, np.arange(len(trace)), bins=bins_2d, range=[[pre_info.plot_low, pre_info.plot_high], [0, hist2d_length]]
        )
        hist_2d += H
    return hist_2d

def summary_statistics(traces: np.ndarray) -> np.ndarray:
    features = []
    for t in tqdm(traces):
        mean = np.mean(t)
        median = np.median(t)
        std = np.std(t)

        x = np.arange(len(t))
        m1 = np.polyfit(x, t, deg=1)
        
        features.append([
            mean,
            median,
            std,
            m1[1],
        ])
    return np.array(features)

def rt_preprocessing(
    traces: np.ndarray, pre_info: PreprocessInfo, longest_cutoff: int, apply_log: bool
) -> Tuple[np.ndarray, np.ndarray]:
    return_data, fullwindow = [], []
    amount_discarded = 0
    for trace in tqdm(traces):
        if apply_log:
            trace = np.log10(trace)
        full = trace.copy()
        full = full[full < pre_info.plot_high]
        full = full[full > pre_info.plot_low]

        trace = trace[trace < pre_info.high]
        trace = trace[trace > pre_info.low]

        if len(trace) < 32:
            amount_discarded += 1
            continue
        if len(trace) > longest_cutoff:
            amount_discarded += 1
            continue
        return_data.append(trace)
        fullwindow.append(full)

    print(f"Amount discarded: {amount_discarded}")
    return np.array(return_data, dtype="object"), np.array(fullwindow, dtype="object")


def extra_features(traces: np.ndarray) -> np.ndarray:
    features = []
    for trace in tqdm(traces):
        length = len(trace)
        mean = np.mean(trace)
        median = np.median(trace)
        std = np.std(trace)

        x = np.arange(len(trace))
        m1 = np.polyfit(x, trace, deg=1)
        rmsd = np.mean((trace - np.polyval(m1, x)) ** 2)

        m2 = np.polyfit(x, trace, deg=2)
        rmsd2 = np.mean((trace - np.polyval(m2, x)) ** 2)

        features.append(
            [
                length,
                mean,
                median,
                std,
                rmsd,
                rmsd2,
                m1[1],
                *m2[2:],
            ]
        )
    return np.array(features)


class ClusteringExperiment:
    def __init__(
        self,
        raw_molecular,
        default_molecular_hists,
        default_blank_hists,
        mea_bins: int,
        rng_seed: int,
        longest_trace: Optional[int] = None,
    ):
        self.raw_molecular = raw_molecular
        self.default_molecular_hists = default_molecular_hists
        self.default_blank_hists = default_blank_hists
        self.rng_seed = rng_seed

        self.mea_bins = mea_bins

        if longest_trace is None:
            longest_trace = 0
            for t in self.raw_molecular:
                if len(t) > longest_trace:
                    longest_trace = len(t)
        self.longest_trace = longest_trace

    def instantiate_models(self, n_clusters: int) -> dict:
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import SpectralClustering
        from sklearn.cluster import KMeans

        gmm_mea = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=self.rng_seed,
        )
        kmeans_mea = KMeans(n_clusters=n_clusters, random_state=self.rng_seed)
        spectral_wenjing = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=self.rng_seed,
        )

        cluster_models = {
            "GMM MEA": gmm_mea,
            "KMeans MEA - mixed": kmeans_mea,
            "KMeans MEA - 28x28": kmeans_mea,
            "Spectral Wenjing - 300": spectral_wenjing,
            "Spectral Wenjing - 900": spectral_wenjing,
        }

        return cluster_models

    def generate_feature_sets(self, low: int, high: int) -> None:
        """
        Generate different feature sets according to litterature:

        tsne_trans
            The 28x28+t-SNE(cos.) feature set with GMM from "Benchmark and application of unsupervised classification approaches for univariate data" DOI: 10.1038/s42005-021-00549-9

        mixed_mea
            XXX

        hist_wenjing
            XXX
        """
        from sklearn.manifold import TSNE

        mea_28x28 = np.array(
            [
                np.histogram2d(
                    trace,
                    np.arange(len(trace)),
                    bins=28,
                    # range=[[low, high], [0, self.longest_trace]],
                    range=[[low, high], [0, 512 + 256]],
                )[0].ravel()
                for trace in self.raw_molecular
            ]
        )

        # we want the old parameters
        tsne = TSNE(n_components=3, metric="cosine", init="random", learning_rate=200.0)
        tsne_trans = tsne.fit_transform(mea_28x28)

        hist_mea = np.array(
            [
                np.histogram(trace, bins=self.mea_bins, range=(low, high))[0]
                for trace in self.raw_molecular
            ]
        )
        mixed_mea = np.concatenate((hist_mea, mea_28x28), axis=1)

        wenjing300 = np.array(
            [
                np.histogram(trace, bins=300, range=(-6, -1))[0]
                for trace in self.raw_molecular
            ]
        )
        wenjing900 = np.array(
            [
                np.histogram(trace, bins=900, range=(-6, -1))[0]
                for trace in self.raw_molecular
            ]
        )

        self.mixed_mea = mixed_mea
        self.mea_28x28 = mea_28x28
        self.wenjing300 = wenjing300
        self.wenjing900 = wenjing900
        self.tsne_trans = tsne_trans

    def plot_2dhist(
        self,
        pred_labels: List[int],
        tunneling_label: int,
        bins_2d: Tuple[int, int],
        plt_low: float,
        plt_high: float,
        ax: Axes,
    ) -> Axes:
        hist_2d = []
        hist_2d = np.zeros(bins_2d)
        for trace in self.raw_molecular[pred_labels == tunneling_label]:
            H, *_ = np.histogram2d(
                trace,
                np.arange(len(trace)),
                bins=bins_2d,
                range=[[plt_low, plt_high], [0, 512]],
            )
            hist_2d += H
        im = ax.imshow(
            hist_2d,
            vmin=0,
            vmax=400,
            origin="lower",
            cmap="viridis",
            extent=[0, 512, plt_low, plt_high],
            aspect="auto",
        )
        return im

    def plot_clustering_results(
        self, cluster_models: dict, plt_low: float, plt_high: float
    ) -> Tuple[Figure, Axes, List[int]]:
        import matplotlib.pyplot as plt
        from sklearn import metrics

        fig, axes = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(14, 24))
        fig.subplots_adjust(hspace=0.05, wspace=0.03)
        ax1_yaxis = np.linspace(
            plt_low, plt_high, self.default_molecular_hists.shape[1]
        )
        max_ax1 = 0
        collected_labels = []
        for idx, (m, ax) in enumerate(
            tqdm(zip(cluster_models.items(), axes.ravel()), total=len(axes.ravel()))
        ):
            x = self.default_molecular_hists
            if m[1].__class__.__name__ == "SpectralClustering":
                if "Wenjing - 300" in m[0]:
                    x = np.corrcoef(self.wenjing300) * 0.5 + 0.5

                if "Wenjing - 900" in m[0]:
                    x = np.corrcoef(self.wenjing900) * 0.5 + 0.5

            if "GMM MEA" in m[0]:
                x = self.tsne_trans

            if "KMeans MEA - mixed" in m[0]:
                x = self.mixed_mea

            if "KMeans MEA - 28x28" in m[0]:
                x = self.mea_28x28

            cluster_labels = m[1].fit_predict(x)
            collected_labels.append(cluster_labels)

            print(f"Analysis from {m[0]}")
            for label in np.unique(cluster_labels):
                pred = self.default_molecular_hists[cluster_labels == label]
                ax.plot(
                    ax1_yaxis,
                    pred.sum(axis=0),
                    label=f"{label + 1}" if idx == 0 else "",
                )
                print(f"Amount of traces in {label + 1}: {len(pred)}")

                if np.max(pred.sum(axis=0)) > max_ax1:
                    max_ax1 = np.max(pred.sum(axis=0))

            ax.plot(
                ax1_yaxis,
                self.default_blank_hists.sum(axis=0),
                c=c[0],
                linestyle="--",
                label="Reference tunneling" if idx == 0 else "",
            )
            ax.plot(
                ax1_yaxis,
                self.default_molecular_hists.sum(axis=0),
                c="k",
                linestyle="--",
                label="Reference molecular" if idx == 0 else "",
            )

            ax.set_xlim(plt_low, plt_high)

        for idx, ax in enumerate(axes.ravel()):
            ax.text(
                -1,
                max_ax1 - max_ax1 * 0.05,
                ALPHABET[idx],
                va="center",
                ha="center",
                weight="bold",
            )
            ax.set_ylim(0, max_ax1 + max_ax1 * 0.05)
            ax.ticklabel_format(axis="y", scilimits=[-5, 4])

        fig.text(
            0.04,
            0.5,
            "Counts",
            va="center",
            ha="center",
            rotation="vertical",
        )
        fig.text(0.5, 0.06, CONDUCTANCE_LABEL, va="center", ha="center")
        axes.ravel()[0].legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(-0.1, 0.93, 1, 0),
            ncol=4,
            columnspacing=1.0,
        )
        return fig, ax, collected_labels