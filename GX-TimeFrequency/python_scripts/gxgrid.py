#!/usr/bin/env python
"""
Script to
"""
from __future__ import annotations

import sys
import os
import shutil
import tempfile
import contextlib
import argparse
import numpy as np
import pandas as pd

from subprocess import Popen, PIPE
from dataclasses import dataclass


def print_df(df, title=None, file=sys.stdout, excel_filename=None) -> None:
    """
    Print entire pandas DataFrame.
    """
    if title is not None: print(title, file=file)
    #with pd.option_context('display.max_columns', None):
    with pd.option_context("display.max_rows", len(df),
                           "display.max_columns", len(list(df.keys())),
                           #"display.precision", precision,
                           ):
        print(df, file=file)
        print("", file=file)

    if excel_filename is not None:
        print("Writing data in excel format to:", excel_filename)
        df.to_excel(excel_filename)


def add_fig_kwargs(func):
    """
    Decorator that adds keyword arguments for functions returning matplotlib figures.

    The function should return either a matplotlib figure or None to signal
    some sort of error/unexpected event.
    See doc string below for the list of supported options.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        # pop the kwds used by the decorator.
        title = kwargs.pop("title", None)
        size_kwargs = kwargs.pop("size_kwargs", None)
        show = kwargs.pop("show", True)
        savefig = kwargs.pop("savefig", None)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", None)
        fig_close = kwargs.pop("fig_close", False)

        # Call func and return immediately if None is returned.
        fig = func(*args, **kwargs)
        if fig is None:
            return fig

        # Operate on matplotlib figure.
        if title is not None:
            fig.suptitle(title)

        if size_kwargs is not None:
            fig.set_size_inches(size_kwargs.pop("w"), size_kwargs.pop("h"), **size_kwargs)

        if ax_grid is not None:
            for ax in fig.axes:
                ax.grid(bool(ax_grid))

        if ax_annotate:
            from string import ascii_letters

            tags = ascii_letters
            if len(fig.axes) > len(tags):
                tags = (1 + len(ascii_letters) // len(fig.axes)) * ascii_letters
            for ax, tag in zip(fig.axes, tags):
                ax.annotate(f"({tag})", xy=(0.05, 0.95), xycoords="axes fraction")

        if tight_layout:
            try:
                fig.tight_layout()
            except Exception as exc:
                # For some unknown reason, this problem shows up only on travis.
                # https://stackoverflow.com/questions/22708888/valueerror-when-using-matplotlib-tight-layout
                print("Ignoring Exception raised by fig.tight_layout\n", str(exc))

        if savefig:
            fig.savefig(savefig)

        import matplotlib.pyplot as plt

        if show:
            plt.show()
        if fig_close:
            plt.close(fig=fig)

        return fig

    # Add docstring to the decorated method.
    s = """\n\n
        Keyword arguments controlling the display of the figure:

        ================  ====================================================
        kwargs            Meaning
        ================  ====================================================
        title             Title of the plot (Default: None).
        show              True to show the figure (default: True).
        savefig           "abc.png" or "abc.eps" to save the figure to a file.
        size_kwargs       Dictionary with options passed to fig.set_size_inches
                          e.g. size_kwargs=dict(w=3, h=4)
        tight_layout      True to call fig.tight_layout (default: False)
        ax_grid           True (False) to add (remove) grid from all axes in fig.
                          Default: None i.e. fig is left unchanged.
        ax_annotate       Add labels to  subplots e.g. (a), (b).
                          Default: False
        fig_close         Close figure. Default: False.
        ================  ====================================================

"""

    if wrapper.__doc__ is not None:
        # Add s at the end of the docstring.
        wrapper.__doc__ += "\n" + s
    else:
        # Use s
        wrapper.__doc__ = s

    return wrapper


def _get_plt(pub_settings=False):
    import matplotlib.pyplot as plt
    if pub_settings:
        font = {'weight': 'normal', 'size': 20}
        plt.rc('font', **font)
        plt.rcParams["mathtext.fontset"] = "cm"
    return plt


@contextlib.contextmanager
def cd_tmpdir():
    """
    Create tmp directory on enter and `cd` to it on enter.
    Remove it and go back to previous dir on exit.
    """
    old_path = os.getcwd()
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(old_path)
        shutil.rmtree(workdir)



@dataclass
class Eratio:
    """
    Small object to store the value of emin, emax and pass it to other procedures.
    """
    emin: float
    emax: float

    @classmethod
    def from_options(cls, options) -> Eratio | None:
        if options.emax is None: return None
        return cls(emin=options.emin, emax=options.emax)

    @classmethod
    def from_eratio(cls, eratio: float) -> Eratio:
        return cls(emin=1, emax=eratio)

    @property
    def r(self) -> float:
        return self.emax / self.emin


@dataclass
class Record:
    ntau: int
    same_list: bool
    tau_erange_list: np.ndarray
    omega_erange_list: np.ndarray


@dataclass
class MinimaxMesh:
    """
    This object stores information on a particular minimax mesh.
    """
    ntau: int
    emin : float
    emax : float
    regterm: float
    taus: np.ndarray
    tau_weights: np.ndarray
    omegas: np.ndarray
    omega_weights: np.ndarray
    costf_tw: np.ndarray
    costf_wt: np.ndarray
    sintf_tw: np.ndarray

    @add_fig_kwargs
    def plot_ft_weights(self, other: MinimaxMesh, self_name="self", other_name="other",
                        with_sinft=False, fontsize=6, **kwargs):
        """
        """
        if self.ntau != other.ntau or self.emin != other.emin or self.emax != other.emax:
            raise ValueError("Cannot compare Minimax meshes with different parameters")

        import matplotlib.pyplot as plt
        nrows, ncols = (4, 2) if with_sinft else (3, 2)
        fig, ax_mat = plt.subplots(nrows=nrows, ncols=ncols,
                                   sharex=False, sharey=False, squeeze=False,
                                   figsize=(12, 8),
                                   #subplot_kw={'xticks': [], 'yticks': []},
        )

        Imat = np.eye(self.ntau)

        select_irow = {
            0: [(self.costf_wt @ self.costf_tw) - Imat,
                (other.costf_wt @ other.costf_tw) - Imat], # , other.costf_wt @ other.costf_tw],
            1: [self.costf_wt, other.costf_wt], # self.costf_wt - other.costf_wt],
            2: [self.costf_tw, other.costf_tw], # self.costf_tw - other.costf_tw],
            3: [self.sintf_tw, other.sintf_tw], # self.sintf_tw - other.sintf_tw],
        }

        label_irow = {
            0: [f"(costf_wt @ costf_tw) - I ({self_name})", f"(costf_wt @ costf_tw) - I ({other_name})"],
            1: [f"costf_wt ({self_name})", f"costf_wt ({other_name})"],
            2: [f"costf_tw ({self_name})", f"costf_tw ({other_name})"],
            3: [f"sintf_tw ({self_name})", f"sintf_tw ({other_name})"],
        }

        for irow in range(nrows):
            for iax, (ax, data, label) in enumerate(zip(ax_mat[irow], select_irow[irow], label_irow[irow])):
                im = ax.matshow(data, cmap='seismic')
                #im = ax.imshow(data, cmap='seismic')
                #fig.colorbar(im, ax=ax, label=label)
                fig.colorbar(im, ax=ax)
                ax.set_title(label, fontsize=fontsize)

        #fig.tight_layout()
        return fig


class GxTabulate:

    def __init__(self, verbose: int):
        self.verbose = verbose

        # Find Fortran executable.
        exec_name = "gx_tabulate_grids.exe"
        self.exec_path = shutil.which(exec_name)
        if self.exec_path is None:
            self.exec_path = shutil.which(os.path.abspath(f"./bin/{exec_name}"))

        if self.exec_path is None:
            raise RuntimeError(f"Cannot find executable: {exec_name}")

        # Call exec_name to get list of eranges for the different ntaus.
        cmd = f"{self.exec_path} eranges"
        with cd_tmpdir():
            if self.verbose > 1: print("Executing cmd:\n", cmd)
            process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True, text=True)
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

        magic_start, magic_end = "<BEGIN ERANGES>", "<END ERANGES>"
        lines = out.splitlines()
        istart, istop = lines.index(magic_start), lines.index(magic_end)
        lines = lines[istart+1:istop]

        # ntau:          14
        # same_list:            1
        # tau_erange_list: 10.000000000000000        15.848900000000000        25.118900000000000 ...
        # omega_erange_list:   10.000000000000000        15.848900000000000        25.118900000000000 ..
        def s2arr(string):
            return np.array([float(s) for s in string.split()])

        name_func = [
            ("ntau", int),
            ("same_list", bool),
            ("tau_erange_list", s2arr),
            ("omega_erange_list", s2arr),
        ]

        magic_start, magic_end = "<BEGIN ERANGE>", "<END ERANGE>"
        self.rec_ntau = {}
        while lines:
            try:
                istart, istop = lines.index(magic_start), lines.index(magic_end)
            except ValueError:
                break
            tokens = lines[istart+1:istop]
            #print(tokens)
            lines = lines[istop+1:]
            d = {}
            for (name, func), l in zip(name_func, tokens):
                beg = name + ":"
                l = l.lstrip()
                assert l.startswith(beg)
                d[name] = func(l.replace(beg, ""))

            ntau = d["ntau"]
            self.rec_ntau[ntau] = Record(**d)

        if verbose > 1:
            print(self)

    def __str__(self) -> str:
        lines = []
        app = lines.append
        app("Info on GreenX Minimax meshes:")
        app(f"Number of meshes: {len(self.rec_ntau)}.")
        for ntau, rec in self.rec_ntau.items():
            app(f"ntau: {ntau}, same_list: {rec.same_list}")
            app(f"tau_erange_list: {rec.tau_erange_list}")
            if not rec.same_list:
                app(f"omega_erange_list: {rec.omega_erange_list}")
        return "\n".join(lines)

    def _exec_print(self, ntau, eratio: Eratio, regterm) -> tuple[dict, MinimaxMesh]:
        """
        """
        rec = self.rec_ntau.get(ntau, None)
        if rec is None:
            raise ValueError(f"ntau {ntau} should be in {list(self.rec_ntau.keys())})")

        cmd = f"{self.exec_path} print -ntau {ntau} -emin {eratio.emin} -emax {eratio.emax} -regterm {regterm}"
        if self.verbose > 1: print("Executing cmd:\n", cmd)

        # Execute executable in a subprocess inside tmpdir.
        with cd_tmpdir():
            process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

            #" mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
            #" max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
            data = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True)
            d = data.iloc[0].to_dict()
            d["ntau"] = ntau

            #  Read FT weigths from dat files ?
            # costf_w_to_t.dat cos(wt) already included. array shape: (nw, nt)
            # costf_t_to_w.dat cos(wt) already included. array shape: (nt, nw)
            # sintf_t_to_w.dat sin(wt) already included. array shape: (nw, nt)
            costf_wt = np.loadtxt("_gx_minimax_print_costf_w_to_t.dat", skiprows=2).T.copy()
            costf_tw = np.loadtxt("_gx_minimax_print_costf_t_to_w.dat", skiprows=2).T.copy()
            sintf_tw = np.loadtxt("_gx_minimax_print_sintf_t_to_w.dat", skiprows=2).T.copy()

        keys = "regterm ntau eratio cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w".split()
        d = {k: d[k] for k in keys}

        mesh = MinimaxMesh(ntau, eratio.emin, eratio.emax, regterm,
                           data.tau.values, data.tau_weight.values, data.omega.values, data.omega_weight.values,
                           costf_wt=costf_wt, costf_tw=costf_tw, sintf_tw=sintf_tw,
                           )

        return d, mesh

    def get_bad_duality_ntau(self, ntau: int, duality_error_tol: float,
                             regterm: float = 0.0) -> pd.DataFrame:
        """
        Return dataframe containing the meshes with `ntau` points
        having duality_error larger than `duality_error_tol`.
        """
        print(f"\nFinding bad minimax meshes with ntau {ntau}, regterm {regterm} and duality error > {duality_error_tol}\n")
        df, meshes = self.get_df_meshes_ntaus(regterm, ntau_list=[ntau])
        return df[df["cosft_duality_error"] > duality_error_tol]

    def optreg_ntau_eratio(self, ntau: int, eratio: Eratio, regterm_list: list[float],
                           plot=True):
        """
        """
        print("\nBegin regfact optimization for ntau:", ntau, "and eratio:", eratio.r, "\n")
        if self.verbose: print("List of regterm values:", regterm_list)

        df, meshes = self.get_df_meshes_ntau_regterms(ntau, eratio, regterm_list)
        best_idx = df["loss_function"].idxmin()
        best_row = df.iloc[best_idx]
        row0 = df.iloc[best_idx]

        if self.verbose:
            if self.verbose > 1:
                print_df(df)
                print("")

            print("Best configuration for row index:", best_idx)
            if best_idx == 0:
                print_df(df.iloc[best_idx])
            else:
                print_df(df.iloc[[0,best_idx]])

        if plot:
            import matplotlib.pyplot as plt
            s = df["regterm"]
            shift = s[s > 0].min() * 1e-1
            df["regterm_shifted"] = df["regterm"] + shift
            column_names = ["cosft_duality_error", "max_err_costf_t_to_w", "max_err_costf_w_to_t", "max_err_sintf_t_to_w"]
            ax_list = df.plot(x="regterm_shifted", y=column_names + ["loss_function"],
                              marker="o", subplots=True, grid=True, logx=True, logy=True)
            ax_list[0].set_title(f"ntau: {ntau}, R: {eratio.r}")
            plt.show()

        return df, meshes

    def get_df_meshes_ntau_regterms(self, ntau: int, eratio: Eratio,
                                    regterm_list: list[float]) -> tuple(pd.DataFrame, list[MinimaxMesh]):
        """
        """
        dicts, meshes = [], []
        for regterm in regterm_list:
            d, mesh = self._exec_print(ntau, eratio, regterm)
            dicts.append(d)
            meshes.append(mesh)

        df = pd.DataFrame(dicts)
        column_names = ["cosft_duality_error", "max_err_costf_t_to_w", "max_err_costf_w_to_t", "max_err_sintf_t_to_w"]
        df["loss_function"] = df[column_names].sum(axis=1)

        return df, meshes

    def get_dataframe_ntaus(self, regterm: float,
                            ntau_list=None) -> tuple(pd.DataFrame, list[MinimaxMesh]):
        """
        """
        if ntau_list is None:
            ntau_list = list(self.rec_ntau.keys())

        dicts, meshes = [], []
        for itau, ntau in enumerate(ntau_list):
            rec = self.rec_ntau[ntau]
            for emax in rec.tau_erange_list:
                eratio = Eratio(emin=1.0, emax=emax - 1e-6)
                d, mesh = self._exec_print(ntau, eratio, regterm)
                dicts.append(d)
                meshes.append(mesh)

        return pd.DataFrame(dicts), meshes

    def get_meshes_eratio_ntaus(self, eratio: Eratio,
                                ntau_list=None) -> list[MinimaxMesh]:
        """
        """
        if ntau_list is None:
            ntau_list = list(self.rec_ntau.keys())

        meshes = []
        for ntau in ntau_list:
            d, mesh = self._exec_print(ntau, eratio, regterm=0.0)
            meshes.append(mesh)

        return meshes

    @add_fig_kwargs
    def plot_err(self, regterm: float,
                 ntau_list=None, what_list=None, figsize=(16, 8), cmap="jet", **kwargs):
        """
        """
        df, meshes = self.get_df_meshes_ntaus(regterm, ntau_list=ntau_list)
        print_df(df)

        plt = _get_plt()

        if what_list is None:
            what_list = [
                "cosft_duality_error",
                "max_err_costf_t_to_w",
                "max_err_costf_w_to_t",
                "max_err_sintf_t_to_w",
            ]

        if len(what_list) % 2 == 0:
            nrows, ncols = len(what_list)//2, 2
        else:
            nrows, ncols = len(what_list), 1

        fig, ax_mat = plt.subplots(nrows=nrows, ncols=ncols,
                                   sharex=True, sharey=False, squeeze=False,
                                   figsize=figsize,
        )

        for iax, ax in enumerate(ax_mat.ravel()):
            attr_name = what_list[iax]
            sc = ax.scatter(df.ntau, df.eratio, s=20,
                            c=np.log10(getattr(df, attr_name)),
                            edgecolors='black', cmap=plt.get_cmap(cmap),
                            #vmax=1e-2,
                            )
            plt.colorbar(sc)

            ax.set_xticks(np.arange(min(df.ntau), max(df.ntau) + 2, 2.0))
            ax.tick_params('both', length=5, width=2, which='major', direction="in")
            ax.tick_params(axis='both', labelsize=20)
            ax.set_yscale('log')
            ax.set_xlabel("number of time/frequency points", fontsize=20)
            ax.set_ylabel(r"$\epsilon_{max}/\epsilon_{min}$", fontsize=30)
            ax.set_title(f'{attr_name} with regterm = {regterm}')

        #plt.savefig('regularization_main.eps', dpi=500)
        return fig

    @add_fig_kwargs
    def plot_grids(self, eratio: Eratio,
                   ntau_list=None, figsize=(16, 8), cmap="jet", **kwargs):
        """
        Plot all the minimax grids for the given eratio.

        Args:

        """
        meshes = self.get_meshes_eratio_ntaus(eratio, ntau_list=ntau_list)

        plt = _get_plt()
        nrows, ncols = 2, 1
        fig, ax_mat = plt.subplots(nrows=nrows, ncols=ncols,
                                   sharex=True, sharey=False, squeeze=False,
                                   figsize=figsize,
        )

        fontsize = 12
        ylabel_iax = {0: "Imaginary frequency (a.u)", 1: "Imaginary time (a.u.)"}
        cmap = plt.get_cmap(cmap)
        mode = "line"

        for iax, ax in enumerate(ax_mat.ravel()):
            for im, mesh in enumerate(meshes):
                ys = mesh.omegas if iax == 0 else mesh.taus
                sizes = mesh.omega_weights if iax == 0 else mesh.tau_weights

                if mode == "line":
                    # Line plot.
                    plot_style = dict(linestyle="-",
                                 marker="o",
                                 color=cmap(im / len(meshes)),
                                 label=f"N={mesh.ntau}",
                                )
                    ax.plot(ys, **plot_style)

                elif mode == "scatter":
                    # Scatter plot with weights.
                    xs = np.arange(mesh.ntau)
                    scatter_style = dict(linestyle="-",
                                 marker="o",
                                 color=cmap(im / len(meshes)),
                                 label=f"N={mesh.ntau}",
                                 s=sizes,
                                 edgecolors='black',
                                )

                    ax.scatter(xs, ys, **scatter_style)

                else:
                    raise ValueError(f"Invalid mode: {mode}")

            if iax == 0:
                ax.set_title(f"R: {eratio.r}")

            ax.set_xlabel("Mesh index", fontsize=fontsize)
            ax.set_ylabel(ylabel_iax[iax], fontsize=fontsize)
            ax.set_yscale('log')
            ax.legend(loc="best", fontsize=8, shadow=True)

        return fig


def gxgrid_bad_duality(options):
    """
    Find minimax grids with large duality error for given ntau.
    """
    g = GxTabulate(verbose=options.verbose)
    df = g.get_bad_duality_ntau(options.ntau, options.duality_error_tol,
                                regterm=options.regterm)
    print_df(df)
    return 0


def gxgrid_optreg(options):
    """
    Optimize regterm for given ntau.
    """
    g = GxTabulate(verbose=options.verbose)

    regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    ntau = options.ntau
    #ntau =  16; emax = 20 # This is very bad

    eratio = Eratio.from_options(options)
    if eratio is not None:
        g.optreg_ntau_eratio(ntau, eratio, regterm_list, plot=True)
        return 0

    # Find bad grids with regterm = 0
    bad_df = g.get_bad_duality_ntau(ntau, options.duality_error_tol)
    print_df(bad_df)

    # Start the brute force optimization of regterm for the bad grids.
    for index, row in bad_df.iterrows():
        emax = row["eratio"] - 1e-6
        eratio = Eratio(emin=1, emax=emax)
        g.optreg_ntau_eratio(ntau, eratio, regterm_list, plot=False)

    return 0


def gxgrid_optreg_low(options):
    """
    Optimize regterm for given ntau and the smallest value of erange.
    """
    g = GxTabulate(verbose=options.verbose)

    ntau = options.ntau
    rec = g.rec_ntau.get(ntau, None)
    if rec is None:
        raise ValueError(f"ntau {ntau} should be in {list(g.rec_ntau.keys())})")

    # Define mesh of erange values
    stop = rec.tau_erange_list[0]
    start = max(10, 1)
    e_samples, step = np.linspace(start, stop, num=10, endpoint=True, retstep=True)
    #e_samples = np.arange(start, stop, step)

    regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    best_rows, origin_rows = [], []

    for ie, emax in enumerate(e_samples):
        eratio = Eratio(emin=1, emax=emax)
        df, meshes = g.optreg_ntau_eratio(ntau, eratio, regterm_list, plot=False)

        # Find the best settings by minimizing the loss_function.
        # TODO:
        # 1) Use duality error as loss function and make sure other errors are smaller than 1e-4
        # 2) Look at other red points
        # 3) Make sure we have the same settings for ntau = 2
        best_idx = df["loss_function"].idxmin()
        best_row = df.iloc[best_idx]
        best_rows.append(best_row)
        origin_rows.append(df.iloc[0])

        if ie == 0:
            # Compare first mesh with optimal one
            meshes[0].plot_ft_weights(meshes[best_idx], self_name="no_regterm", other_name="with_regterm")
            raise ValueError()

    # Build dataframe with best results and original results with regfact = 0.
    data = {
        "eratio": e_samples,
        "regterm": [row.regterm for row in best_rows],
    }

    attr_names = ["cosft_duality_error",
                  "max_err_costf_t_to_w", "max_err_costf_w_to_t", "max_err_sintf_t_to_w",
                 ]

    for aname in attr_names:
        data.update({
            f"old_{aname}": [getattr(row, aname) for row in origin_rows],
            f"best_{aname}": [getattr(row, aname) for row in best_rows],
            })
    df = pd.DataFrame(data)

    excel_filename = f"optreg_low_ntau_{ntau}.xlsx"
    print_df(df, excel_filename=excel_filename)
    os.system(f"open {excel_filename}")

    return 0


def gxgrid_plot_err(options):
    """
    Plot duality error for all minimax grids.
    """
    g = GxTabulate(verbose=options.verbose)
    #ntau_list = [16, 18, 20, 22, 24]
    ntau_list = [6, 8]
    g.plot_err(regterm=0.0, ntau_list=ntau_list)
    return 0


def gxgrid_plot_grids(options):
    """
    Plot minimax grids.
    """
    eratio = Eratio.from_options(options)
    if eratio is None:
        eratio = Eratio(emin=1, emax=100)
        print("Using default eratio:", eratio, "as emax is not specified in input!")

    g = GxTabulate(verbose=options.verbose)
    g.plot_grids(eratio)
    return 0


def get_epilog() -> str:
    s = """\
================================================
Usage example:

gxgrid.py plot_grids --emax 200       --> Plot grids with R = emax/1

================================================
"""
    return s


def get_parser(with_epilog=False) -> argparse.ArgumentParser:

    DUALITY_ERROR_TOL = 0.1
    parser = argparse.ArgumentParser(epilog=get_epilog() if with_epilog else "",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Parent parser for common options.
    copts_parser = argparse.ArgumentParser(add_help=False)
    copts_parser.add_argument('-v', '--verbose', default=0, action='count', # -vv --> verbose=2
        help='verbose, can be supplied multiple times to increase verbosity')
    copts_parser.add_argument('-sns', "--seaborn", const="paper", default=None, action='store', nargs='?', type=str,
        help='Use seaborn settings. Accept value defining context in ("paper", "notebook", "talk", "poster"). Default: paper')
    # Create the parsers for the sub-commands
    subparsers = parser.add_subparsers(dest='command', help='sub-command help',
                                       description="Valid subcommands, use command --help for help")

    def add_opts(p, char_list: list[str]) -> None:
        if "n" in char_list:
            p.add_argument('-n', '--ntau', type=int, required=True, help="Number of mesh points.")
        if "d" in char_list:
            p.add_argument('-d', '--duality-error-tol', type=float, default=DUALITY_ERROR_TOL,
                           help=f"Tolerance on the duality error. Default: {DUALITY_ERROR_TOL}")
        if "r" in char_list:
            p.add_argument('-r', '--regterm', type=float, default=0.0, help="Reguralization term.")
        if "emin" in char_list:
            p.add_argument('--emin', type=float, default=1.0, help="Minimum transition energy (default 1).")
        if "emax" in char_list:
            p.add_argument('--emax', type=float, default=None, help="Maximum transition energy")


    # Parser for optreg command.
    p_optreg = subparsers.add_parser('optreg', parents=[copts_parser], help=gxgrid_optreg.__doc__)
    add_opts(p_optreg, ["n", "d", "emin", "emax"])

    # Parser for optreg_low command.
    p_optreg_low = subparsers.add_parser('optreg_low', parents=[copts_parser], help=gxgrid_optreg_low.__doc__)
    add_opts(p_optreg_low, ["n", "d", "emin", "emax"])

    # Parser for bad_duality command.
    p_bad = subparsers.add_parser('bad_duality', parents=[copts_parser], help=gxgrid_bad_duality.__doc__)
    add_opts(p_bad, ["n", "d", "r"])

    # Parser for plot_err command.
    p_plot_err = subparsers.add_parser('plot_err', parents=[copts_parser], help=gxgrid_plot_err.__doc__)

    # Parser for plot_grids command.
    p_plot_grids = subparsers.add_parser('plot_grids', parents=[copts_parser], help=gxgrid_plot_grids.__doc__)
    add_opts(p_plot_grids, ["emin", "emax"])

    return parser


def main() -> int:

    def show_examples_and_exit(err_msg=None, error_code=1) -> None:
        """Display the usage of the script."""
        sys.stderr.write(get_epilog())
        if err_msg: sys.stderr.write("Fatal Error\n" + err_msg + "\n")
        sys.exit(error_code)

    parser = get_parser(with_epilog=True)

    # Parse command line.
    try:
        options = parser.parse_args()
    except Exception as exc:
        print(exc)
        show_examples_and_exit(error_code=1)

    # Use seaborn settings.
    if getattr(options, "seaborn", None):
        import seaborn as sns
        sns.set(context=options.seaborn, style='darkgrid', palette='deep',
                font='sans-serif', font_scale=1, color_codes=False, rc=None)

    # Dispatch
    return globals()[f"gxgrid_{options.command}"](options)


if __name__ == "__main__":
    sys.exit(main())
