import { pointCoersion } from "@/app/sift/inout";
import { getOsmPart } from "@/app/osm/page";
import { expect, test } from "bun:test";
import SampleBuildings from "./sample_buildings.json";

test("Point Coersion", () => {
  const res = pointCoersion(SampleBuildings as any);
  for (const feat of res.features) {
    expect(feat.geometry.type).toBe("Point");
    expect(feat.geometry.coordinates).toHaveLength(2);
  }
});

test("OSM Parsing", () => {
  const res1 = getOsmPart("I'm sorry, I can't help with that.");
  expect(res1).toBeNull();
  const res2 = getOsmPart(`
    \`\`\`overpassql
    [out:json]
    (
      node["addr:street"]["addr:housenumber"];
      way["addr:street"]["addr:housenumber"];
      relation["addr:street"]["addr:housenumber"];
    );
    \`\`\`
  `);
  expect(res2).toBeDefined();
});
