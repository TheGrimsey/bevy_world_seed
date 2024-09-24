*Placing things like trees & rocks. Should have conditions (don't place above X height, etc). Probably using scenes. Needs a "exclusion zone" thing to stop from spawning in certain areas as well.*

## Requirements
- Allow placing out trees, foliage, rock coverage, etc.
- Deterministic (Ideally possible to identify a feature by their tile & index consistently, no matter if preceding features spawned or not).
- Prevent overlaps with other structures through some for of "exlusion zone".
- High-performant (parallelizable & allow generating large areas without stalling)

## Theoretical Implementations
- "Placement attempts per chunk"
    - Algorithm:
        1. Each feature is given a bounding box/sphere. 
        2. We try to place it
        3. All features with an overlapping bounding box/sphere are discarded.
            - Probably want some bitmask for allowed overlaps. Ex. rocks can overlap with other boulders.
            - Overlaps include "exclusion zones"
            - Might need a different prioritization than just overlaps. Ex. "Keep biggest" to ensure we get some features.
    - Pros:
        - Simple.
        - Works for trees, rocks, and small details.
    - Cons:
        - Requires generating feature placement for sorrounding tiles after regenerating a tile.
        - Feature sizes can't be greater than a tile in size.
        - Does not work for large structures.
        - Expensive when checking intersection of many features
        - May overcorrect and result in no features spawning.
        
