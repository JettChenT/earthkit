# Plan for developing the Open Streetmap query composition feature

Basically, I'm going to build a cursor-ish interface that generate overpass turbo queries.
I might fine-tune starcoder v2 on overpass turbo queries, or perhaps just use GPT for now.

From the OSINT community:

```
Hard-coding a list of common features is one of the challenges with the OSM tool - there are so many more features to use, but it's not feasible for many users to identify the correct feature to add as a custom feature for their search. So some "live suggestions/completions for openstreetmap tags and features" could be really useful
```

** Live Suggestions ** is important for existing users. It would likely help them a lot.
