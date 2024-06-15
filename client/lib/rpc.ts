import { Coords } from "./geo";
import { ProgressDelta } from "./progress_manager";

export type CoordMsg = Coords & { type: "Coords" };
export type ProgressMsg = ProgressDelta & {
  type: "ProgressUpdate";
};

export async function* ingestStream(
  stream: ReadableStream
): AsyncIterable<Msg> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let { value, done } = await reader.read();

  while (!done) {
    const eventString = decoder.decode(value);
    const events = eventString.split("\n\n");
    for (const event of events) {
      if (event === "") continue;
      const res: Msg = JSON.parse(event.slice(6));
      yield res;
    }
    ({ value, done } = await reader.read());
  }
}

export type Msg = CoordMsg | ProgressMsg;
