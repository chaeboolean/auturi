class AuturiTuner:
    def __init__(self, min_num_env: int, max_num_env: int):
        """AuturiTuner Initialization.

        Args:
            min_num_env (int): Minimum number of total environments.
            max_num_env (int): Maximum number of total environments.
        """
        assert min_num_env <= max_num_env
        self.min_num_env = min_num_env
        self.max_num_env = max_num_env

    def next(self):
        pass
