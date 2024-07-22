
# Terrain Deformation with Splines in Bevy.

This uses a spline to flatten terrain (as one would with a road). The terrain heightmap is updated every `500ms`.

## Current issues

- This is slow. Checking every (cached) spline point for every point on the heightmap for every terrain tile.
**Solution**: ??? Optimization??

- Checks the distance to each point and not the distance to the line between points (this kinda works due to subdividing the spline enough but makes it so you have to subdivide it so that there the space between points is < than the width of the spline).
**Solution**: Check the distance to the line between points instead of distance to points directly.