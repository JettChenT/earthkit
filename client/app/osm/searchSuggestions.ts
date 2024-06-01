import type { TypedDocument, Orama, Results, SearchParams } from "@orama/orama";
import { create, insert, insertMultiple, search } from "@orama/orama";
import keys from "@/public/keys.json";
import tags from "@/public/tags.json";

export interface KeyRecord {
  key: string;
  description: string;
  count_all_fraction: number;
}

export interface TagRecord {
  key: string;
  value: string;
  description: string;
  count_all_fraction: number;
}

const t_keys = keys as KeyRecord[];
const t_tags = tags as TagRecord[];

type Document = TypedDocument<Orama<typeof schema>>;

export const schema = {
  name: "string",
  description: "string",
  type: "string",
  meta: {
    osm_frac: "number",
  },
} as const;

export async function initializeDb(): Promise<Orama<typeof schema>> {
  const db: Orama<typeof schema> = await create({
    schema,
  });

  await insertMultiple(
    db,
    t_keys.map((key) => ({
      name: key.key,
      description: key.description,
      type: "key",
      meta: {
        osm_frac: key.count_all_fraction,
      },
    }))
  );

  await insertMultiple(
    db,
    t_tags.map((tag) => ({
      name: `${tag.key}=${tag.value}`,
      description: tag.description,
      type: "tag",
      meta: {
        osm_frac: tag.count_all_fraction,
      },
    }))
  );

  return db;
}

export async function searchDb(
  db: Orama<typeof schema>,
  query: string
): Promise<Results<Document>> {
  const searchParams: SearchParams<Orama<typeof schema>> = {
    term: query,
    limit: 15,
  };
  return search(db, searchParams);
}
