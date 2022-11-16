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
    RESET = "RESET"
    AGGREGATE = "AGGREGATE"
    SEED = "SEED"
    SET_ENV = "SET_ENV"
