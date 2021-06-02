#
# Explicit Runge-Kutta time integrators for PDEs
#

from libpdefd.config_string import config_string 

class ExplicitRungeKutta(config_string):

    def __init__(
            self,
            *wargs,
            **kwargs
    ):
        if len(kwargs) != 0 or len(wargs) != 0:
            self.setup(*wargs, **kwargs)


    def setup(
            self,
            diff_eq_methods,        # Space discretization as well as operators 
            time_integration_order,  # Order to use
            **kwargs
    ):
        self.diff_eq_method = diff_eq_methods
        self.time_integration_order = time_integration_order

        assert self.time_integration_order in [1, 2, 4]


    def get_config_string_array(
            self,
            i_filter_list = []
    ):
        retval = [["ERK"]]
        retval += [["ord", self.time_integration_order]]
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
        if self.time_integration_order == 1:
            retval = i_U + i_dt*self.diff_eq_method.comp_du_dt(i_U, i_timestamp, i_dt)
            return retval

        elif self.time_integration_order == 2:
            S1 = self.diff_eq_method.comp_du_dt(i_U, i_timestamp, 0.5*i_dt)
            S2 = self.diff_eq_method.comp_du_dt(i_U + 0.5 * i_dt * S1, i_timestamp+i_dt, 0.5*i_dt)
            retval = i_U + i_dt*S2
            return retval

        elif self.time_integration_order == 4:
            S1 = self.diff_eq_method.comp_du_dt(i_U, i_timestamp, 1.0/6.0*i_dt)
            S2 = self.diff_eq_method.comp_du_dt(i_U + 0.5 * i_dt * S1, i_timestamp+0.5*i_dt, 1.0/3.0*i_dt)
            S3 = self.diff_eq_method.comp_du_dt(i_U + 0.5 * i_dt * S2, i_timestamp+0.5*i_dt, 1.0/3.0*i_dt)
            S4 = self.diff_eq_method.comp_du_dt(i_U + 1.0 * i_dt * S3, i_timestamp+i_dt, 1.0/6.0*i_dt)
            retval = i_U + i_dt*(1.0/6.0*S1 + 1.0/3.0*S2 + 1.0/3.0*S3 + 1.0/6.0*S4)
            return retval

        else:
            raise Exception("Order "+str(self.time_integration_order)+" not supported")

        return retval
