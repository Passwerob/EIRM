"""EIRM: physically ordered AMI-event-camera simulation.

The implementation package is named ``ami_sim``. EIRM is the public package
and command-line name for the Project simulator.
"""

__project__ = "EIRM"
__version__ = "0.1.0"

from .types import CameraIntrinsics, EventStream

__all__ = ["CameraIntrinsics", "EventStream", "__project__", "__version__"]
