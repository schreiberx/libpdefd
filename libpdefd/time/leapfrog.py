#! /usr/bin/env python3



from libpdefd.config_string import config_string
import libpdefd.time.erk as time_erk
import numpy as np

class Leapfrog(config_string):
    
    def __init__(
            self,
            *wargs,
            **kwargs
    ):
        if len(kwargs) != 0 or len(wargs) != 0:
            self.setup(*wargs, **kwargs)

    def setup(
            self,
            diff_eq_methods,            # Realization of Space discretization as well as operators
            time_integration_order,     # Order to use
            leapfrog_ra_filter_value = None
    ):
        assert time_integration_order == 2
        
        self.diff_eq_method = diff_eq_methods
        self.U_prev = None
        self.U_bar_prev = None
        
        self.robert_asselin_filter_coeff = leapfrog_ra_filter_value
        
        self.erk = time_erk.ExplicitRungeKutta(diff_eq_methods, 4)


    def get_config_string_array(
            self,
            i_filter_list = []
    ):
        retval = [["LF"]]
        return retval


    def comp_time_integration(
            self,
            i_U,
            i_dt,
            i_timestamp
    ):
        """
        :param i_U:  Current state
        :param i_dt: Timestep size
        :param i_timestamp: Current timestamp
        :return: solution at new time step
        """
        
        """
        Use RK4 in case that the previous time step doesn't exist, yet
        """
        if self.U_prev == None and self.U_bar_prev == None:
            U_new = self.erk.comp_time_integration(i_U, i_dt, i_timestamp)
            
            if self.robert_asselin_filter_coeff == None:
                self.U_prev = i_U
            else:
                self.U_bar_prev = i_U
            
            return U_new
        
        if self.robert_asselin_filter_coeff == None:
            """
            Regular leapfrog scheme
            """
            U_next = self.U_prev + 2.0*i_dt*self.diff_eq_method.comp_du_dt(i_U, i_timestamp, 2.0*i_dt)
            
            self.U_prev = np.copy(i_U)
            return U_next
        
        else:
            """
            Use Robert-Asselin filter
            U_prev => U_bar_prev
            """
            U_next = self.U_bar_prev + 2.0*i_dt*self.diff_eq_method.comp_du_dt(i_U, i_timestamp, 2.0*i_dt)
            
            """
            Compute filtered U_bar_current 
            """
            U_bar_current = i_U + 0.5*self.robert_asselin_filter_coeff*(self.U_bar_prev - 2.0*i_U + U_next)
            
            # Not used anymore!
            #self.U_prev = np.copy(i_U)
            self.U_bar_prev = U_bar_current
            return U_next

