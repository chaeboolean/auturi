# This file defines the C++ symbols that need to be exported (aka ABI, application binary interface).
# These symbols will be used by other libraries.
# Note: This file is used for linux only, and should be kept in sync with `ray_api_exported_symbols.lds`.
# Ray ABI is not finalized, the exact set of exported (C/C++) APIs is subject to change.
VERSION_1.0 {
    global:
        # common
        *[0-9]ray[0-9]*;
        TaskExecutionHandler;
        GetFunctionManager;
        GetRemoteFunctions;
        InitRayRuntime;
    local: 
        *;
};
