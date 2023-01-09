import json
from contextlib import contextmanager
import time
import os
import glob
import numpy as np

PROC_PATH="/home/ooffordable/auturi/trace_tmp"
RESULT_PATH="/home/ooffordable/auturi/trace"


def merge_file(output_name):
    pid = 10101
    main_data = []

    input_files = glob.glob(f"{PROC_PATH}/*.json")
    for filename in input_files:
        proc_name = filename.split("/")[-1].split(".")[0]
        with open(filename) as f:
            worker_data = json.load(f)
            for wdata in worker_data:
                wdata["pid"] = pid
                main_data += [wdata]

        os.remove(filename)

    with open(os.path.join(RESULT_PATH, output_name)+".json", "w") as f:
        json.dump(main_data, f)


class EmptyTraceRecorder:
    def __init__(self, proc_name):
        pass
    def __enter__(self, event_name, batch=None):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def dump_stats(self, file_name: str):
        pass

    def clear(self):
        pass

    @contextmanager
    def timespan(self, event_name, batch=None):
        yield None


class ChromeTraceRecorder:
    """This class records events from an event manager and generate trace which can be fed onto Chrome visualizer.
    Notes:
        - After you enable the recorder, it will receive all the following events and store them in its internal buffer.
          Without periodic flush using ``.dump_stats()``, allocated memory size will grows.
    Examples:
        >>> event_manager = EventManager()  # or any object that complies to ``EventManagerProtocol``
        >>> recorder = ChromeTraceRecorder(event_manager)
        >>> with recorder:  # Or, recorder.enable() followed by record.disable() works fine.
        >>>     ... # Do something
        >>> recorder.dump_stats("trace-for-chrome.json")
        >>> # Now open up chrome://tracing in chrome browser and click "Load" button to visualize the output file.
    For more information on format of trace file,
    see https://github.com/kwlzn/pytracing/blob/master/pytracing/pytracing.py
    """
    def __init__(self, proc_name):
        self.proc_name = proc_name
        self.buffer: List[dict] = []

    @contextmanager
    def timespan(self, event_name, batch=None):
        start_time = time.time() * 1e+6 # Timestamp in microseconds
        yield
        end_time = time.time() * 1e+6 
        event = dict(
            tid=self.proc_name,
            name=event_name,  # Event Name.
            ts=start_time,  
            ph="X",
            dur=end_time-start_time,
            batch=batch, 
        )
        self.buffer.append(event)


    def dump_stats(self):
        """Flush all dumps into json file named ``file_name``, after which it clears its internal buffer."""

        file_name = os.path.join(PROC_PATH, self.proc_name) + ".json"
        with open(file_name, "w") as file:
            file.write("[\n")
            for dict_ in self.buffer:

                if "Policy" in self.proc_name:
                    dict_["args"] = dict(batch=dict_["batch"].tolist())
                    dict_.pop("batch")

                json.dump(dict_, file)
                file.write(",\n")
            file.write("{}]\n")  # empty {} so the final entry doesn't end with a comma
        self.buffer.clear()

    def clear(self):
        self.buffer.clear()


def create_tracer(proc_name, to_trace: bool):
    if to_trace:
        return ChromeTraceRecorder(proc_name)
    else:
        return EmptyTraceRecorder(proc_name)
