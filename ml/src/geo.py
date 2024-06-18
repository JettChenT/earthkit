from attr import field, define
from typing import Any, Tuple, Self, Optional
import geopy.distance
from geopy.distance import Distance


@define
class Point:
    lon: float = field()
    lat: float = field()
    aux: dict = field(default={})

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.lon if idx == 0 else self.lat
        elif isinstance(idx, str):
            if idx == "lon":
                return self.lon
            elif idx == "lat":
                return self.lat
            else:
                return self.aux[idx] if self.aux else None

    def lonlat(self):
        return (self.lon, self.lat)

    def latlon(self):
        return (self.lat, self.lon)

    def distance(self, other):
        return geopy.distance.distance(self.latlon(), other.latlon())

    def update_aux(self, **kwargs):
        self.aux.update(kwargs)


@define
class Coords:
    coords: list[Point] = field(factory=list)

    def append(self, point: Point):
        self.coords.append(point)

    def get_bounds(self) -> Optional["Bounds"]:
        if not self.coords:
            return None
        lo = Point(self.coords[0].lon, self.coords[0].lat)
        hi = Point(self.coords[0].lon, self.coords[0].lat)
        for point in self.coords:
            lo.lon = min(lo.lon, point.lon)
            lo.lat = min(lo.lat, point.lat)
            hi.lon = max(hi.lon, point.lon)
            hi.lat = max(hi.lat, point.lat)
        return Bounds(lo, hi)

    def sample(self, bounds: "Bounds", interval: Distance):
        sampled_coords = Coords()
        ul_lat, ul_lon = bounds.lo.lat, bounds.lo.lon
        br_lat, br_lon = bounds.hi.lat, bounds.hi.lon
        width, height = bounds.get_wh()
        w_len, h_len = int(width.km // interval.km), int(height.km // interval.km)
        lat_iv, lon_iv = (br_lat - ul_lat) / h_len, (br_lon - ul_lon) / w_len
        for point in self.coords:
            i, j = (
                int((point.lat - ul_lat) // lat_iv),
                int((point.lon - ul_lon) // lon_iv),
            )
            if 0 <= i < h_len and 0 <= j < w_len:
                sampled_coords.append(point)
        return sampled_coords

    def plot(self, f_out="tmp/panos_plot.html", radius=0.3):
        import pydeck as pdk

        plt_coords = [{"lat": x.lat, "lng": x.lon, "aux":x.aux} for x in self.coords]
        layer = pdk.Layer(
            "ScatterplotLayer",
            plt_coords,
            get_position=["lng", "lat"],
            radius_scale=6,
            get_radius=radius,
            get_fill_color=[255, 140, 0],
            pickable=True
        )

        # Calculate the upper and lower bounds for latitude and longitude
        latitudes = [x.lat for x in self.coords]
        longitudes = [x.lon for x in self.coords]
        pnt_upb = (max(latitudes), max(longitudes))
        pnt_downb = (min(latitudes), min(longitudes))

        view_state = pdk.ViewState(
            latitude=(pnt_upb[0] + pnt_downb[0]) / 2,
            longitude=(pnt_upb[1] + pnt_downb[1]) / 2,
            zoom=15,
        )

        r = pdk.Deck(layers=[layer], initial_view_state=view_state)
        r.to_html(f_out)
        print(f"Plot generated and saved to '{f_out}'.")
    
    def inject_idx(self):
        for i, point in enumerate(self.coords):
            point.update_aux(idx=i)
        return self

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.coords[idx]
        elif isinstance(idx, slice):
            return Coords(self.coords[idx])

    def __iter__(self):
        return iter(self.coords)
    
    def __len__(self):
        return len(self.coords)


@define
class Bounds:
    lo: Point = field()
    hi: Point = field()

    @classmethod
    def from_points(cls, ul: Point, lr: Point):
        lo = Point(min(ul.lon, lr.lon), min(ul.lat, lr.lat))
        hi = Point(max(ul.lon, lr.lon), max(ul.lat, lr.lat))
        return cls(lo, hi)

    def get_wh(self) -> Tuple[Distance, Distance]:
        height = geopy.distance.distance(self.lo.latlon(), (self.hi.lat, self.lo.lon))
        width = geopy.distance.distance(self.lo.latlon(), (self.lo.lat, self.hi.lon))
        return width, height

    def sample(self, interval: Distance) -> Coords:
        width, height = self.get_wh()
        wcnt, hcnt = int(width.km // interval.km), int(height.km // interval.km)
        coords = Coords()
        for i in range(hcnt):
            lat = self.lo.lat + (i / hcnt) * (self.hi.lat - self.lo.lat)
            for j in range(wcnt):
                lon = self.lo.lon + (j / wcnt) * (self.hi.lon - self.lo.lon)
                coords.append(Point(lon, lat))
        return coords


if __name__ == "__main__":
    bounds = Bounds(
        Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
    )
    interval = Distance(kilometers=0.05)
    sampled = bounds.sample(interval)
    sampled.plot()
