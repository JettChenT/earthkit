import { OverpassOptions, OverpassRuntimeError } from "./overpass";
import { OverpassJson } from "./types";

export const humanReadableBytes = (bytes: number) => {
  return bytes > 1024 * 1024
    ? `${Math.round((bytes / (1024 * 1024)) * 100) / 100}MiB`
    : `${Math.round((bytes / 1024) * 100) / 100}KiB`;
};

export const matchAll = (regex: RegExp, string: string) => {
  let match,
    matches = [];
  while ((match = regex.exec(string))) matches.push(match[1]);

  return matches;
};

export const sleep = (ms: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
};

export const consoleMsg = (msg: string) => {
  console.log(`${new Date(Date.now()).toISOString()} Overpass: ${msg}`);
};

// export const oneLessRetry = (opts: OverpassOptions) =>
//   Object.assign({}, opts, {
//     numRetries: opts.numRetries - 1,
//   });

export class OverpassError extends Error {
  constructor(message: string) {
    super(`Overpass Error: ${message}`);
  }
}

export const endpointName = (endpoint: string): string => {
  return new URL(endpoint).hostname;
};

export interface OverpassQuery {
  name: string | null;
  output: "raw" | "json" | "xml" | "csv" | "stream";
  query: string;
  options: OverpassOptions;
}

export const defaultOverpassQuery = {};

export const buildQueryObject = (
  query: string | Partial<OverpassQuery>,
  overwriteObj: {
    name?: string;
    output?: "raw" | "json" | "xml" | "csv" | "stream";
    options?: Partial<OverpassOptions>;
  } = {}
): OverpassQuery => {
  let queryObj;

  // build query object if we just get query string
  if (typeof query === "string")
    queryObj = {
      query: query,
    };
  else if (!("query" in query))
    throw new Error("Query Object must have {query}");
  else queryObj = query;

  // overwrite options
  queryObj = Object.assign(
    {},
    { name: null, output: "raw", options: {} },
    queryObj,
    overwriteObj
  );

  return queryObj as OverpassQuery;
};
