class SHMCommand:
    TERM = 0
    INIT = 1
    INIT_LOOP = 2
    STOP_LOOP = 3
    CMD_DONE = 5


class EnvCommand(SHMCommand):
    RESET = 10
    AGGREGATE = 11
    SEED = 12
    SET_ENV = 13


class PolicyCommand(SHMCommand):
    LOAD_MODEL = "LOAD_MODEL"
    SET_POLICY_ENV = "SET_POLICY_ENV"


class ActorCommand(SHMCommand):
    RECONFIGURE = "RECONFIGURE"
    RUN = "RUN"


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
