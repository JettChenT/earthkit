import { Col, defaultCols } from "./cols";
import { TableEncapsulation, importData } from "./inout";
import { LabelType, TableItem } from "./siftStore";
import SampleDat from "./sample_dat.json";

export const MOCK = false;

const generateDeterministicFeature = (index: number) => {
  return ((index % 10) + 1 + index / 77).toFixed(1);
};

const generateMockItems = (numItems: number): TableItem[] => {
  const coordinatesPool = [
    { lat: 35.6587, lon: 139.4089 },
    { lat: 35.6588, lon: 139.4098 },
    { lat: 35.659, lon: 139.41 },
    { lat: 35.6592, lon: 139.411 },
    { lat: 35.6594, lon: 139.412 },
    { lat: 35.6596, lon: 139.413 },
  ];

  const statuses: LabelType[] = ["Match", "Keep", "Not Match", "Not Labeled"];

  return Array.from({ length: numItems }, (_, index) => {
    const coord = coordinatesPool[index % coordinatesPool.length];
    const status = statuses[index % statuses.length];
    return {
      coord,
      status,
      aux: {
        feata: parseFloat(generateDeterministicFeature(index)),
        featb: parseFloat(generateDeterministicFeature(index + 1)),
        featc: parseFloat(generateDeterministicFeature(index + 2)),
        featd: parseFloat(generateDeterministicFeature(index + 3)),
        feate: parseFloat(generateDeterministicFeature(index + 4)),
        feath: parseFloat(generateDeterministicFeature(index + 5)),
      },
    };
  });
};

const mockTableEncapsulation = SampleDat as TableEncapsulation;

export const mockItems = mockTableEncapsulation.items;

export const mockCols: Col[] = mockTableEncapsulation.cols;
