# EarthKit

[Join Our Discord!](https://discord.gg/X3YRuwZBNn)

A nifty toolkit for geolocation

## Main Features
- **Sift**: An interface for browsing through details about coordinates and enhancing their metadata with AI(including ML models like GeoCLIP, Visual-Place Recognition, as well as Vision-Language Models like gpt-4o). Design inspired by [Elicit](https://elicit.com/).
- **Overpass Query**: An interface for querying [Overpass Turbo](https://overpass-turbo.eu) with natural language as well as OSM feature and location suggestions. We use RAG + in-context learning to generate better queries, following a similar approach as [Overpass NL](https://arxiv.org/pdf/2308.16060). Inspired by [Bellingcat OSM Search](https://osm-search.bellingcat.com/) and [Cursor](https://www.cursor.com/).
- **Geo Estimation**: This runs the SOTA model on [geoestimation](https://paperswithcode.com/task/photo-geolocation-estimation): [GeoClip](https://github.com/VicenteVivan/geo-clip) on user-provided images, and renders a heatmap of predicted results.
- **Satellite & Streetview Geolocation**: (Experimental as the cold-boot time on Modal is too slow) This matches user target imagery with satellite or streetview imager to determine how similar they are. Uses [EigenPlaces](https://github.com/gmberton/EigenPlaces) for streetview geolocation(ground level) and [Sample4Geo](https://github.com/Skyy93/Sample4Geo) for satellite geolocation(crossview).

## General Architecture choices
- Frontend: Next.JS + Vercel
- Database: Redis (will consider postgres for data storage in the future)
- Backend + AI/ML Endpoints: Modal
- Authentication: Clerk

## Codebase Structure

- `client/`: The NextJS codebase of our frontend
- `ml/`: The codebase of our backend + AI/ML endpoints based on Modal and FastAPI. Uses [rye](https://rye-up.com/) for package management.

## Self-Hosting

Earthkit will be self-hostable. I am still working on documenting the self-hosting process for the time being.