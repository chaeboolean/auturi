class SHMCommand:
    INIT = "INIT"
    TERM = "TERM"
    INIT_LOOP = "INIT_LOOP"
    STOP_LOOP = "STOP_LOOP"


class EnvCommand(SHMCommand):
    RESET = "RESET"
    AGGREGATE = "AGGREGATE"
    SEED = "SEED"
    SET_ENV = "SET_ENV"


class PolicyCommand(SHMCommand):
    LOAD_MODEL = "LOAD_MODEL"
    SET_ENV = "SET_ENV"


class PolicyStateEnum:
    STOPPED = 0
    READY = 14
    ASSIGNED = 15
    LOADING_MODEL = 16


class EnvStateEnum:
    """Indicates single simulator state.
    Initialized to STOPPED.

    """

    STOPPED = 0
    STEP_DONE = 22  # Newly arrived requests
    QUEUED = 23  # Inside Server side waiting queue
    POLICY_DONE = 24  # Processed requests
    POLICY_OFFSET = 30  # offset => ASSIGNED
