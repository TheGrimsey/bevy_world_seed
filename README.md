
# World Seed

Tiled runtime terrain generation with modifiers for the [Bevy](https://bevyengine.org/) game engine.

With simplex noise & texturing by rules (or modifiers).

## Supported Version

| Crate Version | Bevy Version | Bevy Lookup Curve |
| ------------- | ------------ | ----------------- |
| 0.1           | 0.14         | 0.5               |

## Collision

This crate does not include colliders as managing different versions of physics crates is a headache. For an example of how to add colliders for crates using Parry3d (Rapier & Avian), see `terrain_collider` example.

## Useful information for procedural generation

- [Henrik Kniberg's Minecraft terrain generation in a nutshell](https://www.youtube.com/watch?v=CSa5O6knuwI)

## TODO

- [ ] Base Height

*Serializable/Deserializable base height that's able to be modified in editor and used instead of (or in addition to) noise.*

### Generation

- [ ] Biome Mapping

*Mapping a place to a biome, the active biome should be usable as a condition for texturing and terrain rules.*

- [ ] Feature Placement (See FEATURE_PLACEMENT.md)

*Placing things like trees & rocks. Should have conditions (don't place above X height, etc). Probably using scenes Needs a "exclusion zone" thing to stop from spawning in certain areas as well.*

### Distant future

- [ ] Switch from normal Mesh to one only containing normals, heights at points, and holes.

*Since all terrains are the same grid, we don't need to recreate the same triangles all the time, only modify the height. This was attempted in the `vertex-format-experiments` branch with minor success*