
# Terrain Deformation with Splines & Shapes in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated when a modifier changes.
 
## Collision

This crate does not include colliders as managing different versions of physics crates is a headache. For an example of how to add colliders for crates using Parry3d (Rapier & Avian), see `terrain_collider` example.

## Useful information for procedural generation

- [Henrik Kniberg's Minecraft terrain generation in a nutshell](https://www.youtube.com/watch?v=CSa5O6knuwI)

## TODO

- [ ] Base Height

*Serializable/Deserializable base height that's able to be modified in editor and used instead of (or in addition to) noise.*

- [ ] Snap To Terrain Height

*Component for snapping an entity to the height of terrain at it's XZ position whenever the tile is rebuilt. Should also include a height offset.*

### Generation

- [ ] Biome Mapping

*Mapping a place to a biome, the active biome should be usable as a condition for texturing and terrain rules.*

- [ ] Feature Placement (See FEATURE_PLACEMENT.md)

*Placing things like trees & rocks. Should have conditions (don't place above X height, etc). Probably using scenes Needs a "exclusion zone" thing to stop from spawning in certain areas as well.*

### Distant future

- [ ] Switch from normal Mesh to one only containing normals, heights at points, and holes.

*Since all terrains are the same grid, we don't need to recreate the same triangles all the time, only modify the height. This was attempted in the `vertex-format-experiments` branch with minor success*