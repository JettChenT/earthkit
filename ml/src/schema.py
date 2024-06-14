from pydantic import BaseModel
from typing import List, Dict, Generic, TypeVar, Optional
from . import geo

AuxT = TypeVar('AuxT')

class Point(BaseModel, Generic[AuxT]):
    lon: float
    lat: float
    aux: Optional[AuxT] = None

    def to_geo(self) -> 'geo.Point':
        return geo.Point(lon=self.lon, lat=self.lat, aux=self.aux)

    @classmethod
    def from_geo(cls, point: 'geo.Point') -> 'Point':
        return cls(lon=point.lon, lat=point.lat, aux=point.aux)

class Coords(BaseModel, Generic[AuxT]):
    coords: List[Point[AuxT]] = []

    def to_geo(self) -> 'geo.Coords':
        return geo.Coords(coords=[p.to_geo() for p in self.coords])

    @classmethod
    def from_geo(cls, coords: 'geo.Coords') -> 'Coords[AuxT]':
        return cls(coords=[Point[AuxT].from_geo(p) for p in coords.coords])

class Bounds(BaseModel, Generic[AuxT]):
    lo: Point[AuxT]
    hi: Point[AuxT]

    def to_geo(self) -> 'geo.Bounds':
        return geo.Bounds.from_points(self.lo.to_geo(), self.hi.to_geo())

    @classmethod
    def from_geo(cls, bounds: 'geo.Bounds') -> 'Bounds[AuxT]':
        return cls(lo=Point[AuxT].from_geo(bounds.lo), hi=Point[AuxT].from_geo(bounds.hi))

if __name__ == "__main__":
    sample_dat = {
        "lat": 37.8267,
        "lon": -122.4233,
        "aux": {
            "speed": 0.0,
            "heading": 0.0,
            "altitude": 0.0,
            "accuracy": 100.0,
            "activity": "still",
        }
    }
    pnt = geo.Point(**sample_dat)
    coords = geo.Coords(coords=[pnt])
    print(Coords.from_geo(coords).model_dump_json())