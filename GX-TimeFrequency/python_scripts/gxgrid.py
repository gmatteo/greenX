#!/usr/bin/env python
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


def print_df(df):
    with pd.option_context('display.max_columns', None):
        print(df)



def _get_plt(pub_settings=False):
    import matplotlib.pyplot as plt
    if pub_settings:
        font = {'weight': 'normal', 'size': 20}
        plt.rc('font', **font)
        plt.rcParams["mathtext.fontset"] = "cm"
    return plt


@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)



@dataclass
class Eratio:
    emin: float
    emax: float

    @classmethod
    def from_options(cls, options) -> Eratio | None:
        if options.emax is None: return None
        return cls(emin=options.emin, emax=options.max)

    @classmethod
    def from_eratio(cls, eratio: float) -> Eratio:
        return cls(emin=1, emax=eratio)

    @property
    def eratio(self):
        return self.emax / self.emin


@dataclass
class Record:
    ntau: int
    same_list: bool
    tau_erange_list: np.ndarray
    omega_erange_list: np.ndarray


@dataclass
class MinimaxMesh:
    ntau: int
    emin : float
    emax : float
    taus: np.ndarray
    tau_weights: np.ndarray
    omegas: np.ndarray
    omega_weights: np.ndarray

    @property
    def eratio(self):
        return self.emax / self.emin



class GxGrid:

    def __init__(self, verbose: int):
        self.verbose = verbose

        # Find executable.
        exec_name = "gx_tabulate_grids.exe"
        self.exec_path = shutil.which(exec_name)
        if self.exec_path is None:
            self.exec_path = shutil.which(os.path.abspath(f"./bin/{exec_name}"))

        if self.exec_path is None:
            raise RuntimeError(f"Cannot find executable: {exec_name}")

        # Call exec_name to get list of eranges for different ntaus.
        cmd = f"{self.exec_path} eranges"
        workdir = tempfile.mkdtemp()
        with cd(workdir):
            if self.verbose: print("Executing cmd:\n", cmd)
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

    def _exec_print(self, ntau, emin, emax, regterm) -> tuple[dict, MinimaxMesh]:
        """
        """
        rec = self.rec_ntau.get(ntau, None)
        if rec is None:
            raise ValueError(f"ntau {ntau} should be in {list(self.rec_ntau.keys())})")

        cmd = f"{self.exec_path} print -ntau {ntau} -emin {emin} -emax {emax} -regterm {regterm}"
        if self.verbose > 1: print("Executing cmd:\n", cmd)

        # Execute executable in a subprocess inside workdir.
        workdir = tempfile.mkdtemp()
        with cd(workdir):
            process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

            #" mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
            #" max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
            data = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True)
            d = data.iloc[0].to_dict()
            d["ntau"] = ntau

        keys = "regterm ntau eratio cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w".split()
        d = {k: d[k] for k in keys}

        mesh = MinimaxMesh(ntau, emin, emax, data.tau.values, data.tau_weight.values,
                           data.omega.values, data.omega_weight.values)

        return d, mesh

    def get_bad_duality_ntau(self, ntau: int, duality_error_tol: float, regterm: float = 0.0) -> pd.DataFrame:
        """
        Return datagrame containing the meshes having duality_error larger
        """
        print(f"\nFinding bad minimax meshes with ntau {ntau}, regterm {regterm} and duality error > {duality_error_tol}\n")
        df = self.get_dataframe_ntaus(regterm, ntau_list=[ntau])
        return df[df["cosft_duality_error"] > duality_error_tol]

    def regopt_ntau_emax(self, ntau: int, emax: float, regterm_list: list[float], plot=True, emin=1.0) -> None:
        """
        """
        print("\nBegin regfact optimization for ntau:", ntau, ", eratio:", emax/emin, "\n")
        if self.verbose: print("List of regterm values:", regterm_list)

        df = self.get_dataframe_ntau_regterms(ntau, emin, emax, regterm_list)
        best_idx = df["loss_function"].idxmin()

        if self.verbose:
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
            ax_list[0].set_title(f"ntau: {ntau}, R: {emax/emin}")
            plt.show()

    def get_dataframe_ntau_regterms(self, ntau, emin, emax, regterm_list):
        """
        """
        dict_list = []
        for regterm in regterm_list:
            d, _mesh = self._exec_print(ntau, emin, emax, regterm)
            dict_list.append(d)

        df = pd.DataFrame(dict_list)
        column_names = ["cosft_duality_error", "max_err_costf_t_to_w", "max_err_costf_w_to_t", "max_err_sintf_t_to_w"]
        df["loss_function"] = df[column_names].sum(axis=1)

        return df

    def get_dataframe_ntaus(self, regterm, ntau_list=None):
        """
        """
        if ntau_list is None:
            ntau_list = list(self.rec_ntau.keys())

        emin = 1.0
        dict_list = []
        for itau, ntau in enumerate(ntau_list):
            rec = self.rec_ntau[ntau]
            for emax in rec.tau_erange_list:
                emax = emax - 1e-6
                d, _mesh = self._exec_print(ntau, emin, emax, regterm)
                dict_list.append(d)

        return pd.DataFrame(dict_list)

    def get_meshes_eminmax_ntaus(self, emin, emax, ntau_list=None) -> list[MinimaxMesh]:
        """
        """
        if ntau_list is None:
            ntau_list = list(self.rec_ntau.keys())

        regterm = 0.0
        meshes = []
        for itau, ntau in enumerate(ntau_list):
            _d, mesh = self._exec_print(ntau, emin, emax, regterm)
            meshes.append(mesh)

        return meshes

    def plot_errors(self, regterm, ntau_list=None, what_list=None, figsize=(16, 8), cmap="jet"):
        """
        """
        df = self.get_dataframe_ntaus(regterm, ntau_list=ntau_list)
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
            c = np.log10(getattr(df, attr_name))

            sc = ax.scatter(df.ntau, df.eratio, c=c, s=20,
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
        plt.show()
        return fig

    def plot_grids(self, emin, emax, ntau_list=None, figsize=(16, 8), cmap="jet"):
        """
        """
        eratio = emax / emin
        meshes = self.get_meshes_eminmax_ntaus(emin, emax, ntau_list=ntau_list)

        plt = _get_plt()
        nrows, ncols = 2, 1
        fig, ax_mat = plt.subplots(nrows=nrows, ncols=ncols,
                                   sharex=True, sharey=False, squeeze=False,
                                   figsize=figsize,
        )

        fontsize = 8
        ylabel_iax = {0: "Frequency points", 1: "Time points"}
        cmap = plt.get_cmap(cmap)

        #print(mesh)
        #mesh.ntau
        #mesh.taus
        #mesh.tau_weights
        #mesh.omegas
        #mesh.omega_weights

        #rcParams['lines.markersize'] ** 2

        for iax, ax in enumerate(ax_mat.ravel()):
            for im, mesh in enumerate(meshes):
                ys = mesh.omegas if iax == 0 else mesh.taus
                sizes = mesh.omega_weights if iax == 0 else mesh.tau_weights

                # Line plot
                plot_style = dict(linestyle="-",
                             marker="o",
                             color=cmap(im / len(meshes)),
                             label=mesh.ntau,
                            )
                ax.plot(ys, **plot_style)

                # Scatter plot with weights
                """
                xs = np.arange(mesh.ntau)
                scatter_style = dict(linestyle="-",
                             marker="o",
                             color=cmap(im / len(meshes)),
                             label=mesh.ntau,
                             #c=c,
                             s=sizes,
                             edgecolors='black',
                            )

                ax.scatter(xs, ys, **scatter_style)
                """

            if iax == 0:
                ax.set_title(f"R: {eratio}")

            ax.set_xlabel("Mesh index", fontsize=fontsize)
            ax.set_ylabel(ylabel_iax[iax], fontsize=fontsize)
            ax.set_yscale('log')
            ax.legend(loc="best", fontsize=fontsize, shadow=True)

        plt.show()

        return fig


def gxgrid_bad_duality(options):
    """
    Find minimax grids with large duality error for given ntau.
    """
    g = GxGrid(verbose=options.verbose)
    df = g.get_bad_duality_ntau(options.ntau, options.duality_error_tol, regterm=options.regterm)
    print_df(df)
    return 0


def gxgrid_regopt(options):
    """
    Optimize regterm for given ntau.
    """
    g = GxGrid(verbose=options.verbose)

    regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    eratio = Eratio.from_options(options)

    #emax = 20
    emax = 100
    emax = 200
    emax = 20
    ntau =  16; emax = 20 # This is very bad
    ntau =  20; emax = 20 # This is very bad
    #ntau =  16; emax = 80 # This is very bad
    #ntau =  24
    #ntau =  20
    g.regopt_ntau_emax(ntau, emax, regterm_list, plot=True)
    return

    ntau = options.ntau

    # Find bad grids with regterm = 0
    bad_df = g.get_bad_duality_ntau(ntau, options.duality_error_tol)
    print_df(bad_df)

    # Start the brute force optimization of regterm for the bad grids.
    plot = False
    for index, row in bad_df.iterrows():
        emax = row["eratio"] - 1e-6
        g.regopt_ntau_emax(ntau, emax, regterm_list, plot=plot)

    #rec = g.rec_ntau[ntau]
    #for emax in rec.tau_erange_list:
    #    emax = emax - 1e-6
    #    g.regopt_ntau_emax(ntau, emax, regterm_list, plot=plot)

    return 0


def gxgrid_plot(options):
    """
    Plot duality error for all grids.
    """
    g = GxGrid(verbose=options.verbose)
    #ntau_list = [16, 18, 20, 22, 24]
    ntau_list = [6, 8]
    regterm = 0.0
    g.plot_errors(regterm, ntau_list=ntau_list)
    return 0


def gxgrid_plot_grids(options):
    """
    Plot grids
    """
    emin = 1
    emax  = 100
    g = GxGrid(verbose=options.verbose)
    g.plot_grids(emin, emax, ntau_list=None)
    return 0



def get_epilog() -> str:
    s = """\
================================================
Usage example:

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

    # Expose option.
    #copts_parser.add_argument('-e', '--expose', action='store_true', default=False,
    #    help="Open file and generate matplotlib figures automatically by calling expose method.")
    #copts_parser.add_argument("-s", "--slide-mode", default=False, action="store_true",
    #    help="Iterate over figures. Expose all figures at once if not given on the CLI.")
    #copts_parser.add_argument("-t", "--slide-timeout", type=int, default=None,
    #    help="Close figure after slide-timeout seconds (only if slide-mode). Block if not specified.")
    copts_parser.add_argument('-sns', "--seaborn", const="paper", default=None, action='store', nargs='?', type=str,
        help='Use seaborn settings. Accept value defining context in ("paper", "notebook", "talk", "poster"). Default: paper')
    # Create the parsers for the sub-commands
    subparsers = parser.add_subparsers(dest='command', help='sub-command help',
        description="Valid subcommands, use command --help for help")

    def add_opts(p, char_list: list[str]) -> None:
        if "n" in char_list:
            p.add_argument('-n', '--ntau', type=int, required=True,
                            help="Number of mesh points.")
        if "d" in char_list:
            p.add_argument('-d', '--duality-error-tol', type=float, default=DUALITY_ERROR_TOL,
                           help=f"Tolerance on the duality error. Default: {DUALITY_ERROR_TOL}")
        if "r" in char_list:
            p.add_argument('-r', '--regterm', type=float, default=0.0, help="Reguralization term.")
        if "emin" in char_list:
            p.add_argument('--emin', type=float, default=1.0, help="Minimum transition energy")
        if "emax" in char_list:
            p.add_argument('--emax', type=float, default=None, help="Maximum transition energy")


    # Parser for regopt command.
    p_opt = subparsers.add_parser('regopt', parents=[copts_parser], help=gxgrid_regopt.__doc__)
    add_opts(p_opt, ("n", "d"))

    # Parser for bad_duality command.
    p_bad = subparsers.add_parser('bad_duality', parents=[copts_parser], help=gxgrid_bad_duality.__doc__)
    add_opts(p_bad, ("n", "d", "r"))

    # Parser for plot command.
    p_plot = subparsers.add_parser('plot', parents=[copts_parser], help=gxgrid_plot.__doc__)

    p_plot_grids = subparsers.add_parser('plot_grids', parents=[copts_parser], help=gxgrid_plot_grids.__doc__)

    return parser


def main():

    def show_examples_and_exit(err_msg=None, error_code=1):
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
    if hasattr(options, "seaborn") and options.seaborn:
        import seaborn as sns
        sns.set(context=options.seaborn, style='darkgrid', palette='deep',
                font='sans-serif', font_scale=1, color_codes=False, rc=None)

    # Dispatch
    return globals()[f"gxgrid_{options.command}"](options)


if __name__ == "__main__":
    sys.exit(main())
