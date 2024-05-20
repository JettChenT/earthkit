# Cropping method for panoramas

Referencing [this issue](https://github.com/gmberton/CosPlace/issues/43)
We do not apply un-distortion to the equirectangular projections.

Current configuration:

- Zoom level: 1
- Panorama cropping: (1024,512) -> (1024, 232) via cropping the top 100 and bottom 180 pixels
- Horizontal cropping: (1024,232) -> 6 \* (232,232) via cropping 6 equal parts, with some overlap
