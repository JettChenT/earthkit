import center from "@turf/center";
import { LabelType, TableItem } from "./siftStore";
import { parse as csvParse } from "csv-parse/sync";
import { stringify } from "csv-stringify/sync";
import { FeatureCollection, Point as GeoJSONPoint } from "geojson";

const parseCsvImport = (input: string): TableItem[] => {
  const records = csvParse(input, {
    columns: true,
  }).map((record: any): TableItem => {
    return {
      coord: {
        lat: record.lat || record.latitude,
        lon: record.lon || record.longitude || record.lng,
      },
      status: record.status,
      aux: {},
    };
  });
  return records;
};

const parseCsvExport = (input: TableItem[]): string => {
  return stringify(
    input.map((item) => ({
      lat: item.coord.lat,
      lon: item.coord.lon,
      status: item.status,
    })),
    {
      header: true,
    }
  );
};

const parseJsonImport = (input: string): TableItem[] => {
  return JSON.parse(input);
};

const parseJsonExport = (items: TableItem[]): string => {
  return JSON.stringify(items);
};

const pointCoersion = (
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
              coordinates: center(f.geometry),
            },
          };
      }
    }) as GeoJSON.Feature<GeoJSON.Point>[],
  };
};

export const parseGeoJsonImport = (
  input: string | FeatureCollection
): TableItem[] => {
  const geoJson =
    typeof input === "string"
      ? (JSON.parse(input) as FeatureCollection)
      : input;
  const pointGeoJson = pointCoersion(geoJson);
  return pointGeoJson.features.map((feature) => ({
    coord: {
      lat: feature.geometry.coordinates[1],
      lon: feature.geometry.coordinates[0],
    },
    panoId: feature.properties?.panoId,
    status: feature.properties?.status || ("Not Labeled" as LabelType),
    aux: {},
  }));
};

const parseGeoJsonExport = (items: TableItem[]): string => {
  const geoJson: FeatureCollection<GeoJSONPoint> = {
    type: "FeatureCollection",
    features: items.map((item) => {
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
  };
  return JSON.stringify(geoJson);
};

export type FormatType = "json" | "geojson" | "csv";

export const importData = (input: string, format: FormatType): TableItem[] => {
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

export const exportData = (items: TableItem[], format: FormatType): string => {
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
