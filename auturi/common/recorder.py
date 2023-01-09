import time
from contextlib import contextmanager
from collections import defaultdict

def make_profiler(init=True):
    class EmptyProfiler:
        @contextmanager
        def timespan(self, name):
            yield
            
        def write(self, filename):
            pass

    class MyProfiler:
        def __init__(self):
            self.ts_dict = defaultdict(list)        
        
        @contextmanager
        def timespan(self, name):
            stime = time.perf_counter()
            yield
            etime = time.perf_counter()
            
            self.ts_dict[name].append(etime - stime)

        def write(self, filename):
            pass

    return MyProfiler() if init else EmptyProfiler()
