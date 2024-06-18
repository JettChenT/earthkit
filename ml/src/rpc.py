from typing import List, Optional, Union, get_args
from attr import define, asdict
from cattrs import Converter
from cattrs.strategies import configure_tagged_union
from typing import Any
from .geo import Coords

@define
class ProgressUpdate:
    facet: str # The facet that is being updated
    total: int # Total number of items that need to be processed
    current: List[str | int] # the id / index of all the items that have been processed
    message: Optional[str] = None # A message to display to the user

@define
class SiftResult:
    idx: int
    value: Any

@define
class ResultsUpdate:
    results: List[SiftResult]

    @classmethod
    def from_coords(cls, coords: Coords, facet: str):
        if 'idx' not in coords.coords[0].aux:
            raise ValueError("Coords must have idx in aux, run inject_idx on the original input")
        return cls(results=[SiftResult(idx=c.aux['idx'], value=c.aux[facet]) for c in coords])
    
    def merge(self, other: "ResultsUpdate") -> "ResultsUpdate":
        return ResultsUpdate(results=self.results + other.results)

converter = Converter()
MsgType = Union[ProgressUpdate, ResultsUpdate]
_MsgCheck = get_args(MsgType)

configure_tagged_union(MsgType, converter, tag_name="type")

def encode_msg(msg: MsgType, enforce: bool = True) -> dict:
    if enforce:
        assert_msg(msg)
    return converter.unstructure(msg, unstructure_as=MsgType)

def sse_encode(msg: MsgType) -> str:
    from .utils import json_encode
    return f"data: {json_encode(encode_msg(msg))}\n\n"

def assert_msg(msg: MsgType) -> None:
    assert isinstance(msg, _MsgCheck), f"Expected a message of type {_MsgCheck}, got {type(msg)}"

if __name__ == "__main__":
    print(encode_msg(ProgressUpdate(facet="test", total=10, current=[1, 2, 3])))
    print(encode_msg(ResultsUpdate(results=[SiftResult(idx=1, value="test"), SiftResult(idx=2, value="test2")])))
