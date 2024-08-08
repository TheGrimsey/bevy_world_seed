
# Terrain Deformation with Splines & Shapes in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated every `500ms`.

## TODO

- [ ] Textures

*Typical terrain texturing where we have a Vec4 where each channel represents a texture's strength.*
    - [ ] From modifiers & heights.

    *Write to it using modifiers, i.e a spline modifier can apply a road texture. Probably want to be able to do some falloff & noise at the edges.*

- [] Take into account the next tile for normals at the edge. (90%, missing corners)

*We get weird edges right now because the smooth normals only account for what exists on the tile.*

- [ ] Holes!

*Terrain needs holes so you can move into caves and stuff.*

- [ ] Base Height

*Serializable/Deserializable base height that's able to be modified in editor and used instead of (or in addition to) noise.*

### Distant future

- [ ] Switch from normal Mesh to one only containing normals, heights at points, and holes.

*Since all terrains are the same grid, we don't need to recreate the same triangles all the time, only modify the height.*