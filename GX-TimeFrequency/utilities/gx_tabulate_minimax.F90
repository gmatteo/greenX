! **************************************************************************************************
!  Copyright (C) 2020-2023 Green-X library
!  This file is distributed under the terms of the APACHE2 License.
!
! **************************************************************************************************
!> Program to print minimax meshes and the associated errors.
!> Use: gx_tabulate_grids.exe -help to list available options.
program gx_tabulate_minimax
#include "gx_common.h"

    use iso_fortran_env, only : std_out => output_unit
    use kinds,           only : dp
    use constants,       only:  ch10
    use gx_minimax,      only : gx_minimax_grid, gx_get_error_message, tau_npoints_supported, &
                                get_points_weights_omega, get_points_weights_tau

    implicit none

    !Local variables-------------------------------
    real(dp), parameter :: zero = 0._dp
    real(dp), parameter :: one = 1._dp
    !scalars
    integer :: ii, it, iw, ierr, num_integ_points, unt, py_unt, nargs, iostat, ntau = -1
    real(dp) :: emin = huge(one), emax = huge(one), eratio, cosft_duality_error, regterm = zero
    real(dp) :: e_range
    character(len=1024) :: msg, arg, arg_val, iomsg, command
    logical :: do_plot = .False., verbose = .False., same_list
    !arrays
    real(dp) :: max_errors(3)
    real(dp), allocatable :: tau_mesh(:), tau_wgs(:), iw_mesh(:), iw_wgs(:)
    real(dp), allocatable :: cosft_wt(:,:), cosft_tw(:,:), sinft_wt(:,:), ac_we(:)
    real(dp), allocatable :: tau_erange_list(:), omega_erange_list(:)

    ! **************************************************************************************************

    nargs = command_argument_count()

    if (nargs < 1) then
        call print_help()
        stop
    end if

    call get_command_argument(1, command)

    do ii=2,nargs
        call get_command_argument(ii, arg)
        if (arg(1:1) /= "-") cycle

        select case (arg)
        case ("-emin")
            call get_command_argument(ii + 1, arg_val)
            read(arg_val, *, iostat=iostat, iomsg=iomsg) emin
            call iocheck()

        case ("-emax")
            call get_command_argument(ii + 1, arg_val)
            read(arg_val, *, iostat=iostat, iomsg=iomsg) emax
            call iocheck()

        case ("-ntau")
            call get_command_argument(ii + 1, arg_val)
            read(arg_val, *, iostat=iostat, iomsg=iomsg) ntau
            call iocheck()

        case ("-regterm")
            call get_command_argument(ii + 1, arg_val)
            read(arg_val, *, iostat=iostat, iomsg=iomsg) regterm
            call iocheck()

        case ("-plot", "-p")
            do_plot = .True.

        case ("-verbose", "-v")
            verbose = .True.

        case ("-h", "--help")
            call print_help()
            stop 0

        case default
            write(std_out, *)"Invalid argument:", trim(arg)
            stop 1
        end select
    end do

    select case (command)
    case ("table")
        call check_emin_emax()
        eratio = emax / emin

        call my_open("_gx_minimax_table.csv", unt)
        write(msg, "(a)") &
                "num_points ierr cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w eratio"
        call dumps(msg, [std_out, unt])

        do ii=1, size(tau_npoints_supported)
            num_integ_points = tau_npoints_supported(ii)

            call gx_minimax_grid(num_integ_points, emin, emax, &
                    tau_mesh, tau_wgs, iw_mesh, iw_wgs, &
                    cosft_wt, cosft_tw, sinft_wt, &
                    max_errors, cosft_duality_error, ierr)

            write(msg, "(2(i0,2x), *(es16.8,2x))") num_integ_points, ierr, cosft_duality_error, max_errors, eratio
            call dumps(msg, [std_out, unt])
        end do
        close(unt)

        ! Write python script to plot the data.
        open(file="_gx_minimax_table.py", newunit=py_unt, action="write")
        write(py_unt, "(13(a,/))") &
                '#!/usr/bin/env python',  &
                'import pandas as pd',  &
                'import matplotlib.pyplot as plt', &
                'df = pd.read_csv("_gx_minimax_table.csv", delim_whitespace=True)', &
                'ynames = ["cosft_duality_error", "max_err_costf_t_to_w",  "max_err_costf_w_to_t",  "max_err_sintf_t_to_w"] # "ierr"', &
                'fig, ax_list = plt.subplots(nrows=len(ynames), ncols=1, sharex=True, squeeze=True)', &
                'for ii, (yname, ax) in enumerate(zip(ynames, ax_list)):', &
                '    if ii == 0:', &
                '       eratio = df["eratio"][0]; max_ierr = df["ierr"].max()', &
                '       fig.suptitle(f"eratio {eratio}, max_ierr: {max_ierr}")', &
                '    df.plot(x="num_points", y=yname, marker="o", ax=ax, logy=True)', &
                '    ax.grid(True)', &
                'plt.show()'
        close(py_unt)

        call execute_command_line("chmod u+x _gx_minimax_table.py")
        write(std_out, "(/,a)")" Execute `./_gx_minimax_table.py` to plot the data with matplotlib or use `-plot` option."
        if (do_plot) call execute_command_line("./_gx_minimax_table.py")

    case ("print")
        call check_emin_emax()
        eratio = emax / emin
        if (ntau == -1) then
            write(std_out, *)"Print command requires ntau"; stop 1
        end if

        call gx_minimax_grid(ntau, emin, emax, &
                tau_mesh, tau_wgs, iw_mesh, iw_wgs, cosft_wt, &
                cosft_tw, sinft_wt, max_errors, cosft_duality_error, ierr, regterm=regterm)

        call handle_gx_ierr(ierr)

        call my_open("_gx_minimax_print_mesh.csv", unt)
        write(msg,"(a)") &
                " mesh_index tau tau_weight omega omega_weight cosft_duality_error max_err_costf_t_to_w"// &
                        " max_err_costf_w_to_t max_err_sintf_t_to_w eratio regterm"
        call dumps(msg, [std_out, unt])
        do it=1,ntau
            write(msg, "(i0, *(es12.5,2x))")it, tau_mesh(it), tau_wgs(it), iw_mesh(it), iw_wgs(it), &
                    cosft_duality_error, max_errors, eratio, regterm
            call dumps(msg, [std_out, unt])
        end do
        close(unt)

        call my_open("_gx_minimax_print_costf_w_to_t.dat", unt)
        write(msg,"(3a,i0)")"# costf_t_to_w. cos(wt) already included. array shape: (nt, nw)",ch10, " ntau: ", ntau
        call dumps(msg, [std_out, unt])
        do it=1,ntau
            write(msg, "(*(es12.5,2x))")cosft_wt(:, it)
            call dumps(msg, [std_out, unt])
        end do
        close(unt)

        call my_open("_gx_minimax_print_costf_t_to_w.dat", unt)
        write(msg,"(3a,i0)")"# costf_w_to_t. cos(wt) already included. array shape: (nw, nt)",ch10, " ntau: ", ntau
        call dumps(msg, [std_out, unt])
        do iw=1,ntau
            write(msg, "(*(es12.5,2x))")cosft_tw(:, iw)
            call dumps(msg, [std_out, unt])
        end do
        close(unt)

        call my_open("_gx_minimax_print_sintf_t_to_w.dat", unt)
        write(msg,"(3a, i0)")" sintf_t_to_w. sin(wt) already included. array shape: (nw, nt) ",ch10, " ntau: ", ntau
        call dumps(msg, [std_out, unt])
        do it=1,ntau
            write(msg, "(*(es12.5,2x))")sinft_wt(:, it)
            call dumps(msg, [std_out, unt])
        end do
        close(unt)

        ! Write python script to plot the data.
        open(file="gx_minimax_print.py", newunit=py_unt, action="write")
        write(py_unt, "(15(a,/))") &
                '#!/usr/bin/env python', &
                'import pandas as pd', &
                'import matplotlib.pyplot as plt', &
                'mesh_df = pd.read_csv("_gx_minimax_print_mesh.csv", delim_whitespace=True)', &
                'title = ""', &
                'for k in "cosft_duality_error max_err_costf_t_to_w max_err_costf_w_to_t max_err_sintf_t_to_w eratio".split():', &
                '    title += f"{k}: {mesh_df[k][0]}\n"', &
                'fig, ax_list = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)', &
                'mesh_df.plot.bar(x="tau", y="tau_weight", ax=ax_list[0])', &
                'mesh_df.plot.bar(x="omega", y="omega_weight", ax=ax_list[1])', &
                '# mesh_df.plot.scatter(x="mesh_index", y="tau", s="tau_weight", ax=ax_list[0])', &
                '# mesh_df.plot.scatter(x="mesh_index", y="omega", s="omega_weight", ax=ax_list[1])', &
                'for ax in ax_list: ax.grid(True)', &
                'fig.suptitle(title, fontsize=8)', &
                'plt.show()'
        close(py_unt)

        call execute_command_line("chmod u+x gx_minimax_print.py")
        write(std_out, "(/,a)")" Execute `./gx_minimax_print.py` to plot the data with matplotlib or use `-plot` option"
        if (do_plot) call execute_command_line("./gx_minimax_print.py")

    case ("eranges")

        write(std_out,"(a)")"<BEGIN ERANGES>"
        do ii=1, size(tau_npoints_supported)
            ntau = tau_npoints_supported(ii)
            allocate(ac_we(2*ntau))
            call get_points_weights_tau(ntau, e_range, ac_we, ierr, tau_erange_list)
            call handle_gx_ierr(ierr)

            call get_points_weights_omega(ntau, e_range, ac_we, ierr, omega_erange_list)
            call handle_gx_ierr(ierr)

            same_list = all(abs(tau_erange_list - omega_erange_list) < 1e-6)
            write(std_out,"(a)")"<BEGIN ERANGE>"
            write(std_out,*)"ntau:", ntau
            write(std_out,*)"same_list: ", merge(1, 0, same_list)
            write(std_out,*)"tau_erange_list:", tau_erange_list(:)
            write(std_out,*)"omega_erange_list:", omega_erange_list(:)
            write(std_out,"(a)")"<END ERANGE>"

            if (.not. same_list) then
                write(std_out,*)"For ntau:", ntau
                write(std_out,*)"erange for omega and tau are not equal!"
                write(std_out,*)"tau_erange_list","omega_erange_list", "abs_diff"
                do it=1,ntau
                 write(std_out,*)tau_erange_list(it), omega_erange_list(it), &
                                 abs(tau_erange_list(it) - omega_erange_list(it))
                end do
                !stop 1
            end if
            deallocate(tau_erange_list, omega_erange_list, ac_we)
        end do
        write(std_out,"(a)")"<END ERANGES>"

    case default
        write(std_out, *)"Invalid command: ", trim(command)
    end select

contains

    subroutine print_help()
        write(std_out, "(a)")"Usage: gx_minimax.x command -emin NUM -emax NUM [-ntau INT] [-plot|-p]"
        write(std_out, "(a,/)")"where allowed commands are: table, print."
        write(std_out, "(a,/)")"The options are as follows:"
        write(std_out, "(a)")  "  -emin   Minimum transition energy."
        write(std_out, "(a)")  "  -emax   Maximum transition energy."
        write(std_out, "(a)")  "  -ntau   Number of mesh-points."
        write(std_out, "(a)")  "  -p      Execute plotting script automatically. "
        write(std_out, "(a)")

        write(std_out, "(a,/)")"To print the mesh with 6 points and erange=emax/emin, use:"
        write(std_out, "(a,/)")"    gx_tabulate_grids.exe print -ntau 6 -emin 0.4 -emax 100"
        write(std_out, "(a,/)")"To print all the meshes with erange=emax/emin, use:"
        write(std_out, "(a,/)")"    gx_tabulate_grids.exe table -emin 0.4 -emax 100"
    end subroutine print_help

    subroutine dumps(string, units)
        character(len=*),intent(in) :: string
        integer,intent(in) :: units(:)
        integer :: jj
        do jj=1,size(units)
            write(units(jj), "(a)") trim(string)
        end do
    end subroutine dumps

    subroutine iocheck()
        if (iostat /= 0) then
            write(std_out, *)"Error while reading argument: ", trim(arg)
            write(std_out, *)"IOMSG: ", trim(iomsg)
            stop 1
        end if
    end subroutine iocheck

    subroutine my_open(filepath, unt)
        character(len=*),intent(in) :: filepath
        integer,intent(out) :: unt

        open(file=filepath, newunit=unt, action="write")
        write(std_out,"(3a)")ch10," Writing data to file: ", trim(filepath), ch10

    end subroutine my_open

    subroutine handle_gx_ierr(ierr)
        integer,intent(in) :: ierr
        if (ierr /= 0) then
            call gx_get_error_message(msg)
            write(std_out, *)trim(msg)
            stop 1
        end if
    end subroutine handle_gx_ierr

    subroutine check_emin_emax()
        if (emin == huge(one) .or. emax == huge(one)) then
            write(std_out,*)"emin and emax arguments are mandatory"; stop 1
        end if

        if (emin <= 1e-6) then
            write(std_out,*)"emin should be greater than zero"; stop 1
        end if
    end subroutine check_emin_emax


end program gx_tabulate_minimax
