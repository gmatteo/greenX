! **************************************************************************************************
!  Copyright (C) 2020-2023 Green-X library
!  This file is distributed under the terms of the APACHE2 License.
!
! **************************************************************************************************
!> Public API for minimax library
!>
!> Compute minimax grid for RPA energy and GW calculation on imaginary time/frequency domain.
!> Usage:
!>
!> ```
!>   call gx_minimax_grid(..., ierr)
!>   if (ierr /= 0) then
!>     call gx_get_error_message(msg)
!>     handle error
!>   end if
!> ```
!>
module gx_minimax
  use minimax_grids,  only : gx_minimax_grid, &
                             gx_minimax_grid_frequency
  use minimax_tau,    only : tau_npoints_supported, get_points_weights_tau
  use minimax_omega,  only : get_points_weights_omega
  use api_utilites,   only: gx_check_ntau, gx_get_error_message
 end module gx_minimax

