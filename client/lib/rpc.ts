import { Coords } from "./geo";
import { ProgressDelta } from "./progress_manager";
import { events, stream } from "fetch-event-stream";

export type CoordMsg = Coords & { type: "Coords" };
export type ProgressMsg = ProgressDelta & {
  type: "ProgressUpdate";
};

export async function* ingestStream(resp: Response): AsyncIterable<Msg> {
  let abort = new AbortController();
  let stream = events(resp, abort.signal);
  for await (const event of stream) {
    if (!event.data) continue;
    const res: Msg = JSON.parse(event.data);
    yield res;
  }
}

export type Msg = CoordMsg | ProgressMsg;
