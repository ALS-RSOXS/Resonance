"""
NEXAFS IO: load from filesystem, accessor for angle filtering and export.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .nexafs_directory import NexafsDirectory


def load_nexafs(
    path: str | Path,
    sample_name: str | None = None,
    formula: str = "C8H8",
    pre_edge: float | tuple[float | None, float | None] = 284.0,
    post_edge: float | tuple[float | None, float | None] = 320.0,
    mode: str = "si_only",
) -> pd.DataFrame:
    """
    Load NEXAFS data from path: discovers files, computes TEY absorption,
    applies Si background subtraction. Pre/post edge bounds are user-configurable.

    Parameters
    ----------
    path : str | Path
        Directory containing NEXAFS scan files (izero*.txt, sample_angle_exp.txt).
    sample_name : str | None
        If set, load only this sample; else load all samples.
    formula : str
        Chemical formula for normalization, e.g. "C8H8".
    pre_edge : float | tuple
        Pre-edge region. Float = upper bound (None, val); tuple = (start, stop).
    post_edge : float | tuple
        Post-edge region. Float = lower bound (val, None); tuple = (start, stop).
    mode : str
        "si_only" or "si_and_oxygen" for background fit.

    Returns
    -------
    pd.DataFrame
        Processed data with Si-subtracted, Density Scaled, Energy, Angle, etc.
        Use df.nexafs.by_angle(55) or df.nexafs.save_parquet(...).
    """
    nd = NexafsDirectory(path)
    df, _ = nd.process_sample(
        sample_name=sample_name,
        formula=formula,
        pre_edge=pre_edge,
        post_edge=post_edge,
        mode=mode,
    )
    return df


@pd.api.extensions.register_dataframe_accessor("nexafs")
class NexafsAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def by_angle(self, angle: float) -> pd.DataFrame:
        """Return rows for the given Angle."""
        if "Angle" not in self._obj.columns:
            raise ValueError("DataFrame has no 'Angle' column")
        return self._obj[self._obj["Angle"] == angle].copy()

    def angles(self) -> list[float]:
        """List unique angles in the data."""
        if "Angle" not in self._obj.columns:
            return []
        return sorted(self._obj["Angle"].dropna().unique().tolist())

    def save_parquet(self, path: str | Path) -> None:
        """Save to parquet file. Requires pyarrow."""
        self._obj.to_parquet(path, index=False)

    def save_csv(self, path: str | Path) -> None:
        """Save to CSV file."""
        self._obj.to_csv(path, index=False)

    def plot(
        self,
        ycol: str = "Si-subtracted",
        ax=None,
        cmap: str = "viridis",
        show_bare_atom: bool = False,
        bare_atom_kwargs=None,
    ):
        """
        Plot NEXAFS spectra with angle-colormap. Requires Energy and Angle columns.

        Parameters
        ----------
        ycol : str
            Column to plot on y-axis (e.g. "Si-subtracted", "Density Scaled", "Mass Abs.").
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        cmap : str
            Colormap name for angle encoding (default "viridis").
        show_bare_atom : bool
            If True, plot Bare Atom Step reference (from first angle).
        bare_atom_kwargs : dict, optional
            kwargs for bare atom line.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if "Energy" not in self._obj.columns:
            raise ValueError("DataFrame has no 'Energy' column")
        if ycol not in self._obj.columns:
            raise ValueError(f"DataFrame has no '{ycol}' column")
        if ax is None:
            _, ax = plt.subplots()
        df = self._obj
        if "Angle" in df.columns:
            angles = sorted(df["Angle"].dropna().unique())
            norm = plt.Normalize(min(angles), max(angles))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            if show_bare_atom and "Bare Atom Step" in df.columns:
                first = df[df["Angle"] == angles[0]]
                plot_kwargs = {"color": "black", "lw": 1}
                if bare_atom_kwargs is not None:
                    plot_kwargs.update(bare_atom_kwargs)
                ax.plot(first["Energy"], first["Bare Atom Step"], **plot_kwargs)
            for angle in angles:
                sub = df[df["Angle"] == angle]
                color = sm.to_rgba(angle)
                ax.plot(sub["Energy"], sub[ycol], label=f"{angle:.1f} deg", color=color)
            ax.legend(title="Angle", handlelength=0.5, fontsize=10, ncol=2, frameon=True, fancybox=False, framealpha=1)
            plt.colorbar(sm, ax=ax, pad=0.02).set_label("Angle (deg)")
        else:
            if show_bare_atom and "Bare Atom Step" in df.columns:
                plot_kwargs = {"color": "black", "lw": 1}
                if bare_atom_kwargs is not None:
                    plot_kwargs.update(bare_atom_kwargs)
                ax.plot(df["Energy"], df["Bare Atom Step"], **plot_kwargs)
            ax.plot(df["Energy"], df[ycol], label=ycol)
        ax.set_xlabel("Energy")
        ax.set_ylabel(ycol)
        sample = df["Sample"].iloc[0] if "Sample" in df.columns else ""
        ax.set_title(f"Sample: {sample}" if sample else "NEXAFS")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        return ax
