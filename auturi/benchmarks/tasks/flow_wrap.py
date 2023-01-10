from flow.utils.registry import env_constructor

def make_env(task_id):
    lib_path = f"auturi.benchmarks.tasks.flow_assets.singleagent.singleagent_{task_id}"
    # module = __import__("flow.benchmarks.{}".format(task_id),
    #                     fromlist=["flow_params"])
    module = __import__(lib_path, fromlist=["flow_params"])

    flow_params = module.flow_params
    return env_constructor(flow_params)

scenarios = [
    # "bottleneck0", 
    # "bottleneck1", 
    # "bottleneck2", 
    # "figureeight0", 
    # "figureeight1", 
    # "figureeight2", 
    # "grid0", 
    # "grid1", 
    # "merge0", 
    # "merge1", 
    # "merge2", 
    "ring", 
    "bottleneck",
    "figure_eight", 
    "merge", 
    "traffic_light_grid", 
]