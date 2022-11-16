class AuturiTuner:
    def __init__(self, min_num_env: int, max_num_env: int, num_collect: int):
        """AuturiTuner Initialization.

        Args:
            min_num_env (int): Minimum number of total environments.
            max_num_env (int): Maximum number of total environments.
            num_collect (int): Total number of trajectories to collect inside loop.

        """
        assert min_num_env <= max_num_env
        self.min_num_env = min_num_env
        self.max_num_env = max_num_env
        self.num_collect = num_collect

    def next(self):
        pass
