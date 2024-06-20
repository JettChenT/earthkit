import { consoleMsg, OverpassError } from "./common";
import "isomorphic-fetch";

export interface ApiStatusOptions {
  verbose: boolean;
}

export const defaultApiStatusOptions: ApiStatusOptions = {
  verbose: false,
};

export const apiStatus = (
  endpoint: string | URL,
  apiStatusOpt: Partial<ApiStatusOptions> = {}
): Promise<OverpassApiStatus> => {
  const opts = Object.assign({}, defaultApiStatusOptions, apiStatusOpt);
  const endpointURL =
    typeof endpoint === "string" ? new URL(endpoint) : endpoint;

  return fetch(endpointURL.href.replace("/interpreter", "/status"))
    .then((resp) => {
      const responseType = resp.headers.get("content-type");

      if (!responseType || responseType.split(";")[0] !== "text/plain")
        throw new OverpassApiStatusError(
          `Response type incorrect (${responseType})`
        );

      return resp.text();
    })
    .then((statusHtml) => {
      const apiStatus = parseApiStatus(statusHtml);

      if (!("clientId" in apiStatus))
        throw new OverpassApiStatusError(`Unable to parse API Status`);

      if (opts.verbose)
        consoleMsg(
          [
            endpointURL.host,
            "status",
            [
              `(rl ${apiStatus.rateLimit}`,
              `sl ${apiStatus.slotsLimited.length}`,
              `sr ${apiStatus.slotsRunning.length})`,
            ].join(" "),
          ].join(" ")
        );

      return apiStatus;
    });
};
export const parseApiStatus = (statusHtml: string): OverpassApiStatus => {
  const status: any = {
    slotsRunning: [],
    slotsLimited: [],
  };

  statusHtml.split("\n").forEach((statusLine) => {
    const lineFirstWord = statusLine.split(" ")[0];
    if (lineFirstWord == "Connected") status["clientId"] = statusLine.slice(14);
    else if (lineFirstWord == "Current")
      status["currentTime"] = statusLine.slice(14);
    else if (lineFirstWord == "Rate")
      status["rateLimit"] = parseInt(statusLine.slice(12));
    else if (lineFirstWord == "Slot")
      status["slotsLimited"].push(
        [statusLine.slice(22).split(", ")].map((splitLine) => ({
          time: splitLine[0],
          seconds: parseInt(splitLine[1].split(" ")[1]),
        }))[0]
      );
    // any lines not "Currently running queries" or "# slots available now"
    // or empty, count those as slots running lines
    else if (
      lineFirstWord != "Currently" &&
      !statusLine.includes("available") &&
      statusLine !== ""
    )
      status["slotsRunning"].push(
        [statusLine.split("\t")].map((splitLine) => ({
          pid: parseInt(splitLine[0]),
          spaceLimit: parseInt(splitLine[1]),
          timeLimit: parseInt(splitLine[2]),
          startTime: splitLine[3],
        }))[0]
      );
  });

  return status;
};

export interface OverpassApiStatus {
  clientId: string;
  currentTime: Date;
  rateLimit: number;
  slotsLimited: OverpassApiStatusSlotLimited[];
  slotsRunning: OverpassApiStatusSlotRunning[];
}

export interface OverpassApiStatusSlotLimited {
  time: string;
  seconds: number;
}

export interface OverpassApiStatusSlotRunning {
  pid: number;
  spaceLimit: number;
  timeLimit: number;
  startTime: string;
}

export class OverpassApiStatusError extends OverpassError {
  constructor(message: string) {
    super(`API Status error: ${message}`);
  }
}
