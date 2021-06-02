


class config_string:
    def __init__(self):
        pass
    
    
    def _merge_tuple(
            self,
            i_tuple,
            i_float_round = 6
    ):
        pretty_config_string = ""

        if len(i_tuple) == 1:
            pretty_config_string += i_tuple[0]
            
        elif len(i_tuple) == 2:

            if isinstance(i_tuple[1], bool):
                pretty_config_string += i_tuple[0]+"="+str(int(i_tuple[1]))

            elif isinstance(i_tuple[1], int):
                pretty_config_string += i_tuple[0]+"="+str(i_tuple[1])

            elif isinstance(i_tuple[1], float):
                pretty_config_string += i_tuple[0]+"="+str(round(i_tuple[1], i_float_round))

            else:
                pretty_config_string += i_tuple[0]+"="+str(i_tuple[1])

        #elif len(i_tuple) == 3:
        #    # Nicer formated config string
        #    pretty_config_string += i_tuple[0]+"="+str(i_tuple[2])
            
        else:
            raise Exception("Tuple error")
    
        return pretty_config_string


    def get_config_string__pretty(
            self,
            i_filter_list = [],
            **kwargs
    ):
        config_array = self.get_config_string_array(i_filter_list)

        # First one should be just a nametag
        assert len(config_array[0]) == 1

        pretty_config_string = config_array[0][0]
        
        for c in config_array[1:]:
            pretty_config_string += " "+self._merge_tuple(c, **kwargs)
        
        return pretty_config_string
    
    
    def get_config_string__nowhitespace(
            self,
            i_filter_list = [],
            **kwargs
    ):
        config_array = self.get_config_string_array(i_filter_list)

        # First one should be just a nametag
        assert len(config_array[0]) == 1

        pretty_config_string = config_array[0][0]
        
        for c in config_array[1:]:
            pretty_config_string += "_"+self._merge_tuple(c, **kwargs)
        
        return pretty_config_string

