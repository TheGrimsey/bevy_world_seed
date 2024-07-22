
# Terrain Deformation with Splines in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated every `500ms`.

## Current issues

- This is slow. Checking every (cached) spline point for every point on the heightmap for every terrain tile.
**Solutions**: ??? Optimization??
[X] Deduplicate points which are too close to each other to contribute significantly.
[ ] Some sort of octree or bounding box to skip points & splines that are too far away.