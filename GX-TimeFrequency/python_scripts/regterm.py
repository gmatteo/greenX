#!/usr/bin/env python
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt', &
import tempfile
import contextlib
import os

@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)


class RegTermScanner:

    def __init__(self, verbose, nprocs=1):
        self.verbose = verbose
        self.nprocs = nprocs
        self.exec_path = os.path.abspath("./bin/gx_tabulate_grids.exe")

        # Get list of eranges
        #workdir = tempfile.mkdtemp()
        #with cd(workdir):
        #    cmd = f"{self.exec_path} eranges"
        #    if self.verbose: print("About to execute command:\n", cmd)
        #    process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)

        #    out, err = process.communicate()
        #    if process.returncode != 0:
        #        raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

    def scan(self, ntau: int, regterm_list: list[float]) -> None:
        """
        Execute the executable in a subprocess inside workdir.
        """
        # Submit the job, return process and pid.
        # ./bin/gx_tabulate_grids.exe print -ntau 6 -emin 0.4 -emax 100
        from subprocess import Popen, PIPE
        emin = 1
        emax = 20

        dict_list = []
        keys = "regterm ntau eratio cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w".split()

        for regterm in regterm_list:
            cmd = f"{self.exec_path} print -ntau {ntau} -emin {emin} -emax {emax} -regterm {regterm}"
            if self.verbose: print("About to execute command:\n", cmd)

            workdir = tempfile.mkdtemp()
            with cd(workdir):
                process = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)

                out, err = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f"Command {cmd}\nreturned {process.returncode}")

                #write(msg,"(a)") &
                #        " mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
                #                " max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
                d = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True).iloc[0].to_dict()
                d["ntau"] = ntau
                dict_list.append({k: d[k] for k in keys})

        df = pd.DataFrame(dict_list)
        column_names = ["cosft_duality_error", "max_err_costf_t_to_w",
                        "max_err_costf_w_to_t", "max_err_sintf_t_to_w"]
        df["loss"]= df[column_names].sum(axis=1)
        best_idx = df['loss'].idxmin()

        with pd.option_context('display.max_columns', None):
            print(df)
            print("")
            print("Best configuration for idx:", best_idx)
            print(df.iloc[best_idx])


    #def plot(self):
    #    """
    #    """


def main():
    """Foo bar"""
    scanner = RegTermScanner(verbose=0)
    ntau = 24
    regterm_list = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    scanner.scan(ntau, regterm_list)
    #scanner.plot()

    return 0



if __name__ == "__main__":
    sys.exit(main())
