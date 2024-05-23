import attr
from typing import List, Optional

# Messages used in Generators and Websockets

@attr.s
class CosSimResults:
    offset: int
    similarity: List[float]

@attr.s
class ProgressUpdate:
    total: int
    current: int
    message: Optional[str] = None
