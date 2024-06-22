import center from "@turf/center";
import { LabelType, TableItem } from "./siftStore";
import { parse as csvParse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";
import { FeatureCollection, Point as GeoJSONPoint } from "geojson";
import { Col } from "./cols";
import { getStats, isnil } from "@/lib/utils";

export type TableEncapsulation = {
  items: TableItem[];
  cols: Col[];
};

type GenericEntries = { [key: string]: any[] };

const deriveColDefs = (entries: GenericEntries): Col[] => {
  return Object.entries(entries).map(([key, vals]) => {
    if (vals.every(Number.isFinite)) {
      return {
        type: "NumericalCol",
        ...getStats(vals),
        accessor: key,
        header: key,
      };
    }
    return { type: "TextCol", accessor: key, header: key };
  });
};
const getEntries = (records: any[]): GenericEntries => {
  const entries: GenericEntries = {};
  records.forEach((record) => {
    Object.entries(record).forEach(([key, value]) => {
      if (!key.startsWith("_")) {
        if (!entries[key]) entries[key] = [];
        entries[key].push(value);
      }
    });
  });
  return entries;
};

const parseCsvImport = (input: string): TableEncapsulation => {
  var auxLst: any[] = [];
  const records = csvParse(input, {
    columns: true,
    cast: (val, ctx) => {
      // if (ctx.index == 0) return val;
      const try_num = Number(val);
      if (Number.isFinite(try_num)) return try_num;
      return val;
    },
  }).map((record: any): TableItem => {
    let { lat, lon, latitude, longitude, lng, status, ...aux } = record;
    auxLst.push(aux);
    return {
      coord: {
        lat: record.lat || record.latitude,
        lon: record.lon || record.longitude || record.lng,
      },
      status: record.status,
      aux,
    };
  });
  return {
    items: records,
    cols: deriveColDefs(getEntries(auxLst)),
  };
};

const enforceString = (value: any) => {
  if (typeof value === "string") return value;
  if (isnil(value)) return "";
  return value.toString();
};

const accessProperty = (value: any, accessor: string) => {
  let parts = accessor.split(".");
  let res = value;
  for (const part of parts) {
    res = res[part];
  }
  return res;
};

const parseCsvExport = (input: TableEncapsulation): string => {
  // Looses the cols information
  // Maybe encode the cols json in base64 comment?? sounds cursed
  return stringify(
    input.items.map((item) => ({
      lat: item.coord.lat,
      lon: item.coord.lon,
      status: item.status,
      ...Object.fromEntries(
        input.cols
          .filter((col) => col.type != "CoordCol" && col.type != "StatusCol")
          .map((col) => [
            col.header,
            enforceString(accessProperty(item.aux, col.accessor)),
          ])
      ),
      raw_auxiliary: item.aux,
    })),
    {
      header: true,
    }
  );
};

const parseJsonImport = (input: string): TableEncapsulation => {
  return JSON.parse(input);
};

const parseJsonExport = (items: TableEncapsulation): string => {
  return JSON.stringify(items);
};

export const pointCoersion = (
  feats: GeoJSON.FeatureCollection
): GeoJSON.FeatureCollection<GeoJSON.Point> => {
  return {
    ...feats,
    features: feats.features.map((f) => {
      switch (f.geometry.type) {
        case "Point":
          return f;
        default:
          return {
            ...f,
            geometry: {
              type: "Point",
              coordinates: center(f.geometry).geometry.coordinates,
            },
          };
      }
    }) as GeoJSON.Feature<GeoJSON.Point>[],
  };
};

export const parseGeoJsonImport = (
  input: string | FeatureCollection
): TableEncapsulation => {
  const geoJson =
    typeof input === "string"
      ? (JSON.parse(input) as FeatureCollection)
      : input;
  const pointGeoJson = pointCoersion(geoJson);
  const auxLst: any[] = [];
  const items = pointGeoJson.features.map((feature) => {
    let { status, aux } = feature.properties as any;
    aux = aux || {};
    auxLst.push(aux);
    return {
      coord: {
        lat: feature.geometry.coordinates[1],
        lon: feature.geometry.coordinates[0],
      },
      status: status || ("Not Labeled" as LabelType),
      aux,
    };
  });
  console.log(auxLst);
  const cols: Col[] =
    (geoJson as any).properties?.cols || deriveColDefs(getEntries(auxLst));
  return { items, cols };
};

const parseGeoJsonExport = (items: TableEncapsulation): string => {
  const geoJson: FeatureCollection<GeoJSONPoint> = {
    type: "FeatureCollection",
    features: items.items.map((item) => {
      const { coord, ...properties } = item;
      return {
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [coord.lon, coord.lat],
        },
        properties,
      };
    }),
    // @ts-ignore : properties are not defined in the geojson type, but we use it to store col defs
    properties: {
      cols: items.cols,
    },
  };
  return JSON.stringify(geoJson);
};

export type FormatType = "json" | "geojson" | "csv";

export const importData = (
  input: string,
  format: FormatType
): TableEncapsulation => {
  switch (format) {
    case "json":
      return parseJsonImport(input);
    case "geojson":
      return parseGeoJsonImport(input);
    case "csv":
      return parseCsvImport(input);
    default:
      throw new Error(`Unsupported format: ${format}`);
  }
};

export const exportData = (
  items: TableEncapsulation,
  format: FormatType
): string => {
  switch (format) {
    case "json":
      return parseJsonExport(items);
    case "geojson":
      return parseGeoJsonExport(items);
    case "csv":
      return parseCsvExport(items);
    default:
      throw new Error(`Unsupported format: ${format}`);
  }
};
