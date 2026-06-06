"""Motor Pool: a grounded diagnostic assistant over public-domain vehicle technical manuals.

Retrieval handles facts. Fine-tuning handles behavior only: citing the source
section, returning structured procedures, and refusing when the manual does not
cover something. Manual knowledge is never stored in model weights.
"""

__version__ = "0.1.0"
