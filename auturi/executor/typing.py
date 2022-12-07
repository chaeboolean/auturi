"""Defines custom types for Auturi Executor."""

from typing import List, NewType, Optional, Tuple, Union

import numpy as np
import ray
import torch.nn as nn

"""
Policy network type.
"""
PolicyModel = NewType("PolicyModel", Optional[nn.Module])

"""
Return type of VectorEnv's poll method
Return np.ndarray directly when remote calls are not necessary.
Else, return env ids for shm backend or ray.ObjectRef for ray backend.
"""
ObservationRefs = NewType(
    "ObservationRefs", Union[np.ndarray, ray.ObjectRef, List[int]]
)


ActionArtifacts = NewType("ActionArtifacts", List[np.ndarray])

ActionTuple = NewType("ActionTuple", Tuple[np.ndarray, ActionArtifacts])


"""
Return type of VectorPolicy's compute_actions method
Return np.ndarray directly when remote calls are not necessary.
Else, return None for shm backend or ray.ObjectRef for ray backend.
"""
ActionRefs = NewType("ActionRefs", Union[np.ndarray, ray.ObjectRef, None])
