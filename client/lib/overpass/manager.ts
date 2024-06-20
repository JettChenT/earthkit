import { main as mainEndpoint } from "./endpoints";
import { OverpassEndpoint } from "./endpoint";
import { OverpassQuery, buildQueryObject } from "./common";
import type { OverpassJson } from "./types";
import type { Readable } from "stream";

export interface OverpassManagerOptions {
  endpoints: string | string[];
  verbose: boolean;
  numRetries: number;
  retryPause: number;
  maxSlots: number;
}

const defaultOverpassManagerOptions = {
  endpoints: mainEndpoint,
  maxSlots: 4,
  numRetries: 1,
  retryPause: 2000,
  verbose: false,
};

export class OverpassManager {
  opts: OverpassManagerOptions = defaultOverpassManagerOptions;
  endpoints: OverpassEndpoint[] = [];
  endpointsInitialized: boolean = false;

  constructor(opts: Partial<OverpassManagerOptions> = {}) {
    this.opts = Object.assign({}, defaultOverpassManagerOptions, opts);
    this.endpoints = [this.opts.endpoints]
      .flat()
      .map(
        (endpointUri) =>
          new OverpassEndpoint(endpointUri, { verbose: this.opts.verbose })
      );
  }

  async _query(
    query: OverpassQuery
  ): Promise<
    Response | OverpassJson | string | Readable | ReadableStream | null
  > {
    if (!this.endpointsInitialized) {
      this.endpointsInitialized = true;
      await Promise.all(
        this.endpoints.map((endpoint) => endpoint.updateStatus())
      );
    }

    return new Promise((res) => {
      const waitForAvailableEndpoint = () => {
        const endpoint = this._getAvailableEndpoint();
        if (endpoint) res(endpoint._query(query));
        else setTimeout(waitForAvailableEndpoint, 100);
      };

      waitForAvailableEndpoint();
    });
  }

  query(query: string | Partial<OverpassQuery>): Promise<Response> {
    return this._query(
      buildQueryObject(query, {
        output: "raw",
      })
    ) as Promise<Response>;
  }

  queryJson(query: string | Partial<OverpassQuery>): Promise<OverpassJson> {
    return this._query(
      buildQueryObject(query, { output: "json" })
    ) as Promise<OverpassJson>;
  }
  queryXml(query: string | Partial<OverpassQuery>): Promise<string> {
    return this._query(
      buildQueryObject(query, { output: "xml" })
    ) as Promise<string>;
  }

  queryCsv(query: string | Partial<OverpassQuery>): Promise<string> {
    return this._query(
      buildQueryObject(query, { output: "csv" })
    ) as Promise<string>;
  }

  queryStream(
    query: string | Partial<OverpassQuery>
  ): Promise<Readable | ReadableStream | null> {
    return this._query(
      buildQueryObject(query, { output: "stream" })
    ) as Promise<Readable | ReadableStream | null>;
  }

  _getAvailableEndpoint(): OverpassEndpoint | null {
    for (let endpoint of this.endpoints) {
      if (endpoint.getSlotsAvailable() > 0) return endpoint;
    }

    return null;
  }
}
