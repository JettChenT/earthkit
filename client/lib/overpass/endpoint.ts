import { apiStatus, OverpassApiStatus } from "./status";
import {
  overpass,
  OverpassGatewayTimeoutError,
  overpassJson,
  OverpassRateLimitError,
  overpassXml,
  overpassCsv,
  overpassStream,
} from "./overpass";
import { OverpassJson } from "./types";
import { OverpassQuery, consoleMsg, sleep, buildQueryObject } from "./common";
import type { Readable } from "stream";

interface OverpassEndpointOptions {
  gatewayTimeoutPause: number;
  rateLimitPause: number;
  minRequestInterval: number;
  maxSlots: number;
  verbose: boolean;
}

const defaultOverpassEndpointOptions = {
  gatewayTimeoutPause: 2000,
  rateLimitPause: 5000,
  minRequestInterval: 2000,
  verbose: false,
  maxSlots: 4,
};

export class OverpassEndpoint {
  statusTimeout: NodeJS.Timeout | null = null;
  status: OverpassApiStatus | null | false = null;
  statusAvailable: boolean = true;
  opts: OverpassEndpointOptions;
  queue: OverpassQuery[] = [];
  queueIndex: number = 0;
  queueRunning: number = 0;
  uri: URL;

  constructor(uri: string, opts: Partial<OverpassEndpointOptions> = {}) {
    this.opts = Object.assign({}, defaultOverpassEndpointOptions, opts);
    this.uri = new URL(uri);
  }

  updateStatus() {
    // clear status timeout it already exists
    if (this.statusTimeout) {
      clearTimeout(this.statusTimeout);
      this.statusTimeout = null;
    }

    return apiStatus(this.uri.href, { verbose: this.opts.verbose })
      .then((apiStatus) => {
        this.status = apiStatus;

        // if there's any rate limited slots and something in the queue
        // set timeout to update status once the rate limit is over
        if (
          this.status.slotsLimited.length == this.getRateLimit() //&&
          // (this.queueIndex < this.queue.length || this.queue.length == 0)
        ) {
          const lowestRateLimitSeconds =
            Math.min(...this.status.slotsLimited.map((slot) => slot.seconds)) +
            0.1;

          if (
            this.opts.verbose &&
            this.status.slotsLimited.length == this.getRateLimit()
          )
            consoleMsg(
              `${this.uri.host} rate limited; waiting ${lowestRateLimitSeconds}s`
            );

          this.statusTimeout = setTimeout(async () => {
            await this.updateStatus();
          }, lowestRateLimitSeconds * 1000);
        }
      })
      .catch((error) => {
        // silently error apiStatus (some endpoints don't support /api/status)
        if (this.opts.verbose)
          consoleMsg(
            `${this.uri.host} ERROR getting api status (${error.message})`
          );

        // set status to false if status endpoint broken
        // make sure we don't ask again
        this.statusAvailable = false;
      });
  }

  async _query(
    queryObj: OverpassQuery
  ): Promise<
    Response | OverpassJson | string | Readable | ReadableStream | null
  > {
    // add query to queue
    const queryIdx = this.queue.push(queryObj);

    // if no name specified in query, use the queue index as name
    if (!queryObj.name) {
      queryObj.name = (queryIdx - 1).toString();
    }

    if (this.opts.verbose)
      consoleMsg(`${this.uri.host} query ${queryObj.name} queued`);

    // initialize endpoint status if endpoint is idle (no queries in queue)
    if (
      !this.status &&
      this.statusAvailable &&
      this.queue.length - 1 == this.queueIndex
    )
      await this.updateStatus();

    // poll queue until a slot is open and then execute query
    return new Promise((res) => {
      const waitForQueue = () => {
        const slotsAvailable = this.getSlotsAvailable();
        if (queryIdx <= this.queueIndex + slotsAvailable) {
          this.queueIndex++;
          this.queueRunning++;
          setTimeout(() => {
            res(this._sendQuery(queryObj));
          }, (this.queueIndex + slotsAvailable - queryIdx - 1) * this.opts.minRequestInterval);
        } else setTimeout(waitForQueue, 100);
      };

      waitForQueue();
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

  _sendQuery(query: OverpassQuery): Promise<Response> {
    if (this.opts.verbose)
      consoleMsg(`${this.uri.host} query ${query.name} sending`);

    // choose overpass function to use based upon desired output
    // allows for checking of runtime errors early in promise chain

    let overpassFn;

    if (query.output == "json") overpassFn = overpassJson;
    else if (query.output == "xml") overpassFn = overpassXml;
    else if (query.output == "csv") overpassFn = overpassCsv;
    else if (query.output == "stream") overpassFn = overpassStream;
    else overpassFn = overpass;

    return overpassFn(query.query, {
      ...query.options,
      endpoint: this.uri.href,
    })
      .then(async (resp: any) => {
        if (this.opts.verbose)
          consoleMsg(`${this.uri.host} query ${query.name} complete`);

        this.queueRunning--;

        if (this.statusAvailable) {
          // if query isn't last one in queue, update status
          //if (this.queueIndex < this.queue.length)
          await this.updateStatus();
          // if query is last, set status = null
          // so a fresh status will be requested if new queries performed
          //else this.status = null;
        }

        return resp;
      })
      .catch(async (error: any) => {
        if (error instanceof OverpassRateLimitError) {
          // if query is rate limited, poll until we get slot available
          if (this.opts.verbose)
            consoleMsg(`${this.uri.host} query ${query.name} rate limited`);

          return new Promise(async (res) => {
            const waitForRateLimit = () => {
              // +1 to account for slotsAvailable not accounting for this
              // particular rate limited request
              if (this.getSlotsAvailable() + 1 > 0) res(this._sendQuery(query));
              else setTimeout(waitForRateLimit, 100);
            };

            if (this.statusAvailable) {
              await this.updateStatus();
            } else await sleep(this.opts.rateLimitPause);

            waitForRateLimit();
          });
        } else if (error instanceof OverpassGatewayTimeoutError) {
          // if query is gateway timeout, pause some ms and send again
          if (this.opts.verbose)
            consoleMsg(`${this.uri.host} query ${query.name} gateway timeout`);

          return sleep(this.opts.gatewayTimeoutPause).then(() =>
            this._sendQuery(query)
          );
        } else {
          // if is other error throw it to be handled downstream

          if (this.opts.verbose)
            consoleMsg(
              `${this.uri.host} query ${query.name} uncaught error (${error.message})`
            );

          this.queueRunning--;

          if (!this.statusTimeout && this.statusAvailable)
            await this.updateStatus();

          throw error;
        }
      });
  }

  getRateLimit(): number {
    if (this.status) {
      // if status is loaded and has limited rate limit return that
      if (this.status.rateLimit > 0) return this.status.rateLimit;
      // if unlimited rate limit, return default maxSlots
      else return this.opts.maxSlots;
    } else {
      // if status is unavailable return default maxSlots
      if (!this.statusAvailable) return this.opts.maxSlots;
      // if status isn't loaded but still available (initial load) return 0
      else return 0;
    }
  }

  getSlotsAvailable(): number {
    const rateLimit = this.getRateLimit();

    if (this.status) {
      // include slotsLimited in available calculation if there's nothing
      // running in the queue (happens on startup)
      return rateLimit - this.queueRunning - this.status.slotsLimited.length;
    } else {
      // if status isn't loaded but still available (initial load) return 0
      if (this.statusAvailable) return 0;
      // if endpoint has broken /api/status just don't include limited slots
      else return rateLimit - this.queueRunning;
    }
  }
}
