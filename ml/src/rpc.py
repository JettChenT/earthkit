from typing import List, Optional, Union, get_args
from attr import define, asdict
from cattrs import Converter
from cattrs.strategies import configure_tagged_union
from .geo import Coords

@define
class ProgressUpdate:
    facet: str # The facet that is being updated
    total: int # Total number of items that need to be processed
    current: List[str | int] # the id / index of all the items that have been processed
    message: Optional[str] = None # A message to display to the user

converter = Converter()
MsgType = Union[ProgressUpdate, Coords]
_MsgCheck = get_args(MsgType)

configure_tagged_union(MsgType, converter, tag_name="type")

def encode_msg(msg: MsgType, enforce: bool = True) -> dict:
    if enforce:
        assert_msg(msg)
    return converter.unstructure(msg, unstructure_as=MsgType)

def assert_msg(msg: MsgType) -> None:
    assert isinstance(msg, _MsgCheck), f"Expected a message of type {_MsgCheck}, got {type(msg)}"

if __name__ == "__main__":
    print(encode_msg(ProgressUpdate(facet="test", total=10, current=[1, 2, 3])))
