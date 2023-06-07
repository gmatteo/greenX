#!/usr/bin/env python
from __future__ import annotations

import sys
import os
import shutil
import tempfile
import contextlib
import numpy as np
import pandas as pd

from subprocess import Popen, PIPE
from dataclasses import dataclass



@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)


@dataclass
class Record:
    ntau: int
    same_list: bool
    tau_erange_list: np.ndarray
    omega_erange_list: np.ndarray


class RegTermOptimizer:

    def __init__(self, verbose):
        self.verbose = verbose

        exec_name = "gx_tabulate_grids.exe"
        self.exec_path = shutil.which(exec_name)
        if self.exec_path is None:
            self.exec_path = shutil.which(os.path.abspath(f"./bin/{exec_name}"))

        if self.exec_path is None:
            raise RuntimeError(f"Cannot find executable: {exec_name}")

        # Call exec_name to get list of eranges for the different ntau.
        cmd = f"{self.exec_path} eranges"
        workdir = tempfile.mkdtemp()
        with cd(workdir):
            if self.verbose: print("About to execute command:\n", cmd)
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


    def _exec_print(self, ntau, emin, emax, regterm) -> dict:
        cmd = f"{self.exec_path} print -ntau {ntau} -emin {emin} -emax {emax} -regterm {regterm}"
        if self.verbose: print("About to execute command:\n", cmd)

        # Execute the executable in a subprocess inside workdir.
        workdir = tempfile.mkdtemp()
        with cd(workdir):
            process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

            #" mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
            #" max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
            d = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True).iloc[0].to_dict()
            d["ntau"] = ntau

        keys = "regterm ntau eratio cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w".split()
        return {k: d[k] for k in keys}

    def optimize_ntau_emax(self, ntau: int, emax: float, regterm_list: list[float], plot=True, emin=1.0) -> None:
        """
        """
        print("")
        print("Begin regfact optimization for ntau:", ntau, "eratio:", emax/emin)
        print("List of regterm values:", regterm_list)

        rec = self.rec_ntau.get(ntau, None)
        if rec is None:
            raise ValueError(f"ntau {ntau} should be in {list(self.rec_ntau.keys())})")

        dict_list = []
        #keys = "regterm ntau eratio cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w".split()

        for regterm in regterm_list:
            d = self._exec_print(ntau, emin, emax, regterm)
            #cmd = f"{self.exec_path} print -ntau {ntau} -emin {emin} -emax {emax} -regterm {regterm}"
            #if self.verbose: print("About to execute command:\n", cmd)

            ## Execute the executable in a subprocess inside workdir.
            #workdir = tempfile.mkdtemp()
            #with cd(workdir):
            #    process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)
            #    out, err = process.communicate()
            #    if process.returncode != 0:
            #        raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

            #    #" mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
            #    #" max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
            #    d = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True).iloc[0].to_dict()
            #    d["ntau"] = ntau
            #dict_list.append({k: d[k] for k in keys})
            dict_list.append(d)

        df = pd.DataFrame(dict_list)
        column_names = ["cosft_duality_error", "max_err_costf_t_to_w", "max_err_costf_w_to_t", "max_err_sintf_t_to_w"]
        df["loss_function"] = df[column_names].sum(axis=1)
        best_idx = df["loss_function"].idxmin()
        #best_row = df.iloc[best_idx]
        #row0 = df.iloc[0]

        with pd.option_context('display.max_columns', None):
            if self.verbose:
                print(df)
                print("")

            print("Best configuration for row index:", best_idx)
            if best_idx == 0:
                print(df.iloc[best_idx])
            else:
                print(df.iloc[[0,best_idx]])

        if plot:
            import matplotlib.pyplot as plt
            s = df["regterm"]
            shift = s[s > 0].min() * 1e-1
            #print("shift", shift)
            df["regterm_shifted"] = df["regterm"] + shift
            ax_list = df.plot(x="regterm_shifted", y=column_names + ["loss_function"],
                              marker="o", subplots=True, grid=True, logx=True, logy=True)
            plt.show()

    def plot_errors(self):
        """
        """
        ntau_list = list(self.rec_ntau.keys())

        emin = 1.0
        regterm = 0.0
        dict_list = []
        for itau, ntau in enumerate(ntau_list):
            rec = self.rec_ntau[ntau]
            #if ntau > 8: break
            for emax in rec.tau_erange_list:
                emax = emax - 1e-6
                d = self._exec_print(ntau, emin, emax, regterm)
                dict_list.append(d)
        df = pd.DataFrame(dict_list)

        with pd.option_context('display.max_columns', None):
            print(df)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16,8))
        font = {'weight': 'normal', 'size': 20}
        plt.rc('font', **font)
        plt.rcParams["mathtext.fontset"] = "cm"

        ax = plt.subplot(111)
        ax.tick_params('both', length=5, width=2, which='major', direction="in")
        ax.tick_params(axis='both', labelsize=20)
        ax.set_yscale('log')

        #file_name = "data_greenx_regularization_0.001"
        #file_name = "data_greenx_regularization_0.01"
        #data = np.loadtxt(file_name, unpack=True)
        #x = data[0]
        #y = data[6]
        #z = data[2]
        x = df.ntau
        y = df.eratio
        z = df.cosft_duality_error

        cmap = plt.get_cmap("jet")

        plt.xticks(np.arange(min(x), max(x)+2, 2.0))

        sc = ax.scatter(x, y,
            edgecolors='black',
            c=np.log10(z),
            s=20,
            cmap=cmap,
            #vmax=1e-2
        )

        ax.set_xlabel("number of time/frequency points", fontsize=20)
        ax.set_ylabel(r"$\epsilon_{max}/\epsilon_{min}$", fontsize=30)
        ax.set_title(f'regularization = {regterm}')
        plt.colorbar(sc)
        #plt.savefig('regularization_main.eps', dpi=500)
        plt.show()
        return

        from scipy.ndimage import uniform_filter1d

        font = {'weight' : 'normal', 'size' : 20}
        plt.rc('font', **font)
        plt.rcParams["mathtext.fontset"] = "cm"
        # plt.rcParams['figure.figsize'] = 1.5,2
        fig = plt.figure(figsize=(12,12))
        ax = plt.subplot(111,aspect = 'equal')
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #plt.xlim(0, 10**9)
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=10)
        #left = 0.15

        #ax = fig.add_axes([left])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params('both', length=20, width=2, which='major', direction="in")
        #ax.tick_params('both', length=10, width=1, which='minor', direction="in",)
        number_of_points = range(6,36,2)

        for i in range(len(number_of_points)):
            file_name = "%s_error_jan" % number_of_points[i]
            data = np.loadtxt(file_name, unpack=True)
            x = data[0]
            y = data[1]
            #kr = KernelReg(y, x, 'c')
            ax.plot(x, y, label='npts = %s' % number_of_points[i])
            ax.text(x[-1], y[-1],f'%s ' % number_of_points[i], fontsize=20)

            ax.set_xlabel(r"$\epsilon_{max}/\epsilon_{min}$", fontsize=35)
            ax.set_ylabel("Minimax error", fontsize=25)
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        fig.tight_layout()

        plt.savefig('err.png', dpi=500)
        plt.show()


def regterm_scan(options):
    """
    regterm_scan
    """
    o = RegTermOptimizer(verbose=options.verbose)
    ntau = 24
    #ntau = 10
    #regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    #regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7]
    ntau = options.ntau
    rec = o.rec_ntau[ntau]

    #rec.tau_erange_list[0]
    for emax in rec.tau_erange_list:
        emax = emax - 1e-6
        o.optimize_ntau_emax(ntau, emax, regterm_list, plot=False)

    return 0


def regterm_plot(options):
    """
    regterm_plot
    """
    o = RegTermOptimizer(verbose=options.verbose)
    o.plot_errors()
    return 0


def get_epilog() -> str:
    s = """\
======================================================================================================
Usage example:

======================================================================================================
"""
    return s


def get_parser(with_epilog=False):

    import argparse
    parser = argparse.ArgumentParser(epilog=get_epilog(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Parent parser for common options.
    copts_parser = argparse.ArgumentParser(add_help=False)
    copts_parser.add_argument('-v', '--verbose', default=0, action='count', # -vv --> verbose=2
        help='verbose, can be supplied multiple times to increase verbosity')

    # Create the parsers for the sub-commands
    subparsers = parser.add_subparsers(dest='command', help='sub-command help',
        description="Valid subcommands, use command --help for help")

    p_scan = subparsers.add_parser('scan', parents=[copts_parser], help=regterm_scan.__doc__)
    p_scan.add_argument('-n', '--ntau', type=int, required=True)

    p_plot = subparsers.add_parser('plot', parents=[copts_parser], help=regterm_plot.__doc__)

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
    #if hasattr(options, "seaborn") and options.seaborn:
    #    import seaborn as sns
    #    sns.set(context=options.seaborn, style='darkgrid', palette='deep',
    #            font='sans-serif', font_scale=1, color_codes=False, rc=None)

    return globals()[f"regterm_{options.command}"](options)

if __name__ == "__main__":
    sys.exit(main())
