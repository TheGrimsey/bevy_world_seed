
# Terrain Deformation with Splines & Shapes in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated every `500ms`.

## Current issues

- This is slow. Checking every (cached) spline point for every point on the heightmap for every terrain tile.
**Solutions**: ??? Optimization??
[X] Deduplicate points which are too close to each other to contribute significantly.
[ ] Some sort of octree or bounding box to skip points & splines that are too far away.

## TODO

- [ ] Textures
*Typical terrain texturing where we have a Vec4 where each channel represents a texture's strength.*
    - [ ] From modifiers & heights.
    *Write to it using modifiers, i.e a spline modifier can apply a road texture. Probably want to be able to do some falloff & noise at the edges.*

- [ ] Take into account the next tile for normals at the edge.
*We get weird edges right now because the smooth normals only account for what exists on the tile.*

- [ ] Dirty Terrain Tiles map.
*Like in OxiNav, append a tile id when it is dirtied (by a shape or otherwise) & heights need to be updated.*