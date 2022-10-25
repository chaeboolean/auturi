class VectorMixin:
    """Mixin used for VectorXX class.

    Provides utility functions to handle one local and multiple remote
    inner components.
    """

    @property
    def current_counter(self):
        return 1 + len(self.remote_workers)

    def add(self, remote_fn):
        pass

    def call_workers(self, total_num: int):
        yield self.local_worker
        for i, worker in self.remote_workers.items():
            yield worker
            if i >= total_num:
                break
