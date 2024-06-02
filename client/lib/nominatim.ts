export type OSMResultT = {
  place_id: number;
  osm_type: string;
  class: string;
  type: string;
  osm_id: number;
  name: string;
  display_name: string;
  address_type: string;
  area_id: number;
};

export const geoSearch = async (query: string) => {
  const response = await fetch(
    `https://nominatim.openstreetmap.org/search?q=${query}&format=json`
  );
  let data: OSMResultT[] = await response.json();
  for (let i = 0; i < data.length; i++) {
    data[i].area_id = data[i].osm_id;
    if (data[i].osm_type == "way") {
      data[i].area_id += 2400000000;
    }
    if (data[i].osm_type == "relation") {
      data[i].area_id += 3600000000;
    }
  }
  return data;
};
