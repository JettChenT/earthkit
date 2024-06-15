export type ProgressBase = {
  facet: string;
  total: number;
};

export type ProgressDelta = ProgressBase & {
  current: number[] | string[];
};

export type ProgressAggr = ProgressBase & {
  progress: number;
};

export interface ProgressManager {}
