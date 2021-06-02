


class TimeIntegrators:
    
    def __init__(
            self,
            **kwargs
    ):
        self.setup(**kwargs)


    def setup(
            self,
            time_integration_method,
            **kwargs,
    ):
        """
        :param time_integration_method: Class of time integration method, e.g. 'ExplicitRungeKutta'
        :param kwargs: Arguments to be forwarded to particular time integration method
        """

        if time_integration_method == "erk":
            import libpdefd.time.erk as erk
            self.time_integration_method = erk.ExplicitRungeKutta(**kwargs)
            return
        
        if time_integration_method == "leapfrog":
            import libpdefd.time.leapfrog as leapfrog
            self.time_integration_method = leapfrog.Leapfrog(**kwargs)
            return

        
        raise Exception("Unknown time integration method '"+time_integration_method+"'")
    
    def get_method(self):
        return self.time_integration_method
