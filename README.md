
# Terrain Deformation with Splines & Shapes in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated when a modifier changes.

## Collision

This crate does not include colliders as managing different versions of physics crates is a headache.

## TODO

- [X] Textures

*Typical terrain texturing where we have a Vec4 where each channel represents a texture's strength.*

    - [X] From modifiers
    - [X] & heights.
    - [X] angle

    *Write to it using modifiers, i.e a spline modifier can apply a road texture. Probably want to be able to do some falloff & noise at the edges.*

- [X] Take into account the next tile for normals at the edge. (We currently ignore corners but it seems to eb fine anyway?)

*We get weird edges right now because the smooth normals only account for what exists on the tile.*

- [ ] Base Height

*Serializable/Deserializable base height that's able to be modified in editor and used instead of (or in addition to) noise.*

- [ ] Snap To Terrain Height

*Component for snapping an entity to the height of terrain at it's XZ position whenever the tile is rebuilt. Should also include a height offset.*

### Modifiers

- [ ] Modifier strength multiplier
*Additional component that allows you to affect the strength, i.e reduce the strength so that instead of pulling to match modifiers it goes 50% toward modifier, flattening the area without making it completely flat.* 

- [ ] Holes!

*Terrain needs holes so you can move into caves and stuff.*

### Distant future

- [ ] Switch from normal Mesh to one only containing normals, heights at points, and holes.

*Since all terrains are the same grid, we don't need to recreate the same triangles all the time, only modify the height.*