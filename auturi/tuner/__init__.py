import auturi.tuner.typing as types

class AuturiTuner(object):
    """
    Online tuning algorithm.
    Input: , Output: 
    It records ...


    """
    def __init__(self, num_sim):
        self.recorder = None
        self.N = num_sim

    # works like generator
    def next_config(self):
        # TODO: Only a
        return types.SamplerConfig(
            bs=self.N,
            numc=1,
        )
 