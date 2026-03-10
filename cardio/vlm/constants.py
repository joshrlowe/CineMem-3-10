"""Special tokens for CardioVLM memory invocation.

Legacy STM/LTM tokens (short-term / long-term memory) are retained for
backward compatibility with CineMem checkpoints.  The dual-memory TDM/PSM
tokens (Transient Dynamics Memory / Persistent Structure Memory) support
the new architecture where transient haemodynamic context is separated
from persistent anatomical structure memory.
"""

from __future__ import annotations

# Legacy STM/LTM tokens
SHORT_INVOKE = "<ms_I>"
SHORT_END = "<ms_E>"
LONG_INVOKE = "<ml_I>"
LONG_END = "<ml_E>"

# Dual-memory TDM/PSM tokens
TDM_INVOKE = "<tdm_I>"
TDM_END = "<tdm_E>"
PSM_INVOKE = "<psm_I>"
PSM_END = "<psm_E>"

ALL_SPECIAL_TOKENS = [
    SHORT_INVOKE,
    SHORT_END,
    LONG_INVOKE,
    LONG_END,
    TDM_INVOKE,
    TDM_END,
    PSM_INVOKE,
    PSM_END,
]
