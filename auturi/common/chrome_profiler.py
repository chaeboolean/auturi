import json
from contextlib import contextmanager
import time
import os
import glob
import numpy as np
from typing import List
from pathlib import Path

PROC_PATH="/workspace/auturi/trace_tmp"
RESULT_PATH="/workspace/auturi/trace/"

def merge_file(out_dir, output_name):
    result_dir = Path(RESULT_PATH) / Path(out_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    merge_json_file(result_dir, output_name)
    a,b =merge_idle_file(result_dir, output_name)
    return a, b

def merge_idle_file(out_dir, output_name):
    input_files = glob.glob(f"{PROC_PATH}/*.txt")
    policy_exec, env_exec = [], []

    def _arr_to_mean(arr):
        return np.mean(np.array(arr))

    with open(os.path.join(out_dir, output_name)+".txt", "w") as outfile:
        for filename in input_files:
            proc_name = filename.split("/")[-1].split(".")[0]
            with open(filename) as f:
                l = f.readline().strip()
                exec, total = l.split(" ") # str
                exec, total = float(exec), float(total)

            ratio = exec/total
            if "EnvProc" in proc_name:
                env_exec.append(ratio)
            else:
                policy_exec.append(ratio)

            outfile.write(f"{proc_name}:  {exec}/{total} = {round(ratio,2)}\n")
            os.remove(filename)


        outfile.write(f"Final Policy: {_arr_to_mean(policy_exec)}\n")
        outfile.write(f"Final Env: {_arr_to_mean(env_exec)}\n")
        return _arr_to_mean(policy_exec), _arr_to_mean(env_exec)


def merge_json_file(out_dir, output_name):
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

    with open(os.path.join(out_dir, output_name)+".json", "w") as f:
        json.dump(main_data, f)


class EmptyTraceRecorder:
    def __init__(self, proc_name):
        pass
    def __enter__(self, event_name, batch=None):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start_loop(self):
        pass
    
    def stop_loop(self):
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
        self._start_loop_ts, self._stop_loop_ts = -1, -1
        self._execution_cnt = 0

    def start_loop(self):
        self._start_loop_ts = time.time() * 1e+6 
        self._execution_cnt = 0
        self.buffer.clear()

    def stop_loop(self):
        self._stop_loop_ts = time.time() * 1e+6 


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
        self._execution_cnt += (end_time-start_time)


    def dump_stats(self):
        """Flush all dumps into json file named ``file_name``, after which it clears its internal buffer."""
        Path(PROC_PATH).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(PROC_PATH, self.proc_name) 
        with open(file_name + ".txt", "w") as f:
            f.write(f"{self._execution_cnt} {self._stop_loop_ts - self._start_loop_ts}\n")

        with open(file_name + ".json", "w") as file:
            file.write("[\n")
            for dict_ in self.buffer:
                if "Policy" in self.proc_name:
                    dict_["args"] = dict(batch=dict_["batch"].tolist())
                    dict_.pop("batch")

                json.dump(dict_, file)
                file.write(",\n")
            file.write("{}]\n")  # empty {} so the final entry doesn't end with a comma
        self.buffer.clear()


def create_tracer(proc_name):
    to_trace = os.getenv("AUTURI_TRACE", False)
    if to_trace:
        return ChromeTraceRecorder(proc_name)
    else:
        return EmptyTraceRecorder(proc_name)
