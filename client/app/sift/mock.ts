import { Col, defaultCols } from "./cols";
import { TableEncapsulation, importData } from "./inout";
import { LabelType, TableItem } from "./siftStore";

export const MOCK = false;

const NUM_ITEMS = 20;

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
export const mockItems = generateMockItems(NUM_ITEMS);

export const mockCols: Col[] = [
  ...defaultCols,
  {
    accessor: "feata",
    header: "Feature A",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
  {
    accessor: "featb",
    header: "Feature B",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
  {
    accessor: "featc",
    header: "Feature C",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
  {
    accessor: "featd",
    header: "Feature D",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
  {
    accessor: "feate",
    header: "Feature E",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
  {
    accessor: "feath",
    header: "Feature H",
    type: "NumericalCol",
    mean: 3.0,
    stdev: 1.0,
  },
];
