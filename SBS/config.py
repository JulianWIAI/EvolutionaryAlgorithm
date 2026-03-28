"""
config.py — Central configuration for the Smart Swarm simulation.

All tunable constants live here so the rest of the codebase stays free of
hard-coded magic numbers. Adjusting population size, mutation rate, maze
layout, or visual style requires changes only in this file.
"""

# ---------------------------------------------------------------------------
# Window & Rendering
# ---------------------------------------------------------------------------
WINDOW_TITLE  = "Smart Swarm — Evolutionary Algorithm Showcase"
WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 600
FPS           = 150         # 2.5x faster than original 60 — keeps museum visitors engaged

# ---------------------------------------------------------------------------
# Evolutionary Algorithm Parameters
# ---------------------------------------------------------------------------
POPULATION_SIZE        = 50   # Number of ants alive simultaneously each generation
# DNA_LENGTH must be large enough for the slalom path (~470 steps minimum for an efficient
# genome; winding early-generation paths can easily need 600+). 800 gives a safe margin.
DNA_LENGTH             = 800  # Genome length = max movement steps per generation
NUM_PARENTS_MATING     = 15   # How many top ants breed the next generation
# Mutation rate is deliberately set to give ~40 gene replacements per offspring
# (5 % × 800 = 40).  The earlier working version used 8 % × 500 = 40 mutations;
# keeping the absolute count the same preserves the probability that any 30-gene
# "wall-navigation block" survives unchanged in ~28 % of offspring, giving ~10
# potentially-improving offspring per generation.  At 12 % (96 mutations) only
# ~0.6 offspring per generation preserved a wall-A block while exploring wall B —
# too low for reliable convergence within a museum-showcase time frame.
MUTATION_PERCENT_GENES = 5    # Percentage of genes randomly replaced each generation
CROSSOVER_TYPE         = "single_point"  # Recombination strategy (kept for reference)

# ---------------------------------------------------------------------------
# Agent (Ant) Physics
# ---------------------------------------------------------------------------
ANT_RADIUS        = 5    # Collision + visual radius in pixels
ANT_SPEED         = 5.0  # Increased from 3.0 — agents traverse maze ~1.7x faster
ANT_MAX_ANGLE_DELTA = 25 # Max heading change in degrees per gene (gene ∈ [-1,1])

# ---------------------------------------------------------------------------
# Environment Layout
# ---------------------------------------------------------------------------
START_X = 60
START_Y = WINDOW_HEIGHT // 2    # Vertical centre of the play area
TARGET_X = WINDOW_WIDTH - 60
TARGET_Y = WINDOW_HEIGHT // 2
# TARGET_RADIUS was 28, which is small relative to ANT_SPEED=3.0. After the full
# slalom, ants must land within this radius. 35 px gives ~11 steps of margin on
# approach, reducing false-negative win detections on oblique trajectories.
TARGET_RADIUS = 35              # Radius of the goal circle in pixels

WALL_THICKNESS = 25

# X-coordinates that mark when each slalom wall has been fully cleared.
# These are the right edge of each wall + a small safety margin so the ant's
# body (radius=5) is fully past the wall before the milestone is awarded.
# Used by the fitness function to create hard score cliffs that force the EA
# to discover each wall gap rather than looping in front of a wall face.
WALL_A_CLEAR_X = 242   # Past wall A  (x=210, width=25 → right edge x=235)
WALL_B_CLEAR_X = 462   # Past wall B  (x=430, width=25 → right edge x=455)
WALL_C_CLEAR_X = 682   # Past wall C  (x=650, width=25 → right edge x=675)

# Maze wall definitions as (x, y, width, height) tuples.
# The three internal hanging walls form a slalom that ants must learn to navigate:
#   Wall A hangs from the TOP  → ants must route BELOW it  (y > 335)
#   Wall B rises from the BOTTOM → ants must route ABOVE it (y < 265)
#   Wall C hangs from the TOP  → ants must route BELOW it  (y > 335)
MAZE_WALLS = [
    # Outer boundary — hard edges that instantly kill any ant that touches them
    (0,                          0,                           WINDOW_WIDTH, WALL_THICKNESS),
    (0,  WINDOW_HEIGHT - WALL_THICKNESS, WINDOW_WIDTH,        WALL_THICKNESS),
    # Slalom wall A: hangs from top, gap is at the bottom
    (210, WALL_THICKNESS,        WALL_THICKNESS, 310),
    # Slalom wall B: rises from bottom, gap is at the top
    (430, WINDOW_HEIGHT - WALL_THICKNESS - 310, WALL_THICKNESS, 310),
    # Slalom wall C: hangs from top, gap is at the bottom
    (650, WALL_THICKNESS,        WALL_THICKNESS, 310),
]

# ---------------------------------------------------------------------------
# Color Palette  — Sci-Fi HUD theme
# ---------------------------------------------------------------------------
COLOR_BG           = (23,  42,  70)    # Rich navy #172A46
COLOR_WALL         = (74,  98, 138)    # Bright icy slate-blue #4A628A
COLOR_WALL_EDGE    = (110, 135, 175)   # Lighter edge accent
COLOR_START_ZONE   = (0,  230,  80)    # Neon green
COLOR_TARGET_ZONE  = (255, 215,  0)    # Bright gold
COLOR_ANT_ALIVE    = (0,  229, 255)    # Neon cyan #00E5FF
COLOR_ANT_DEAD     = (40,  30,  50)    # Very dim — fades into background
COLOR_ANT_BEST     = (255, 230,  50)   # Gold for the confirmed winner/best
COLOR_TRAIL_ALIVE  = (0,  150, 200)    # Dim cyan (SRCALPHA base colour)
COLOR_TRAIL_BEST   = (220, 160,  20)   # Dim gold (SRCALPHA base colour)
# Uniform colours used during the active simulation phase.
# All agents — alive and dead — share these colours so no ant appears to be
# "winning" before fitness has actually been evaluated at generation's end.
COLOR_ANT_UNIFORM   = (0,  200, 240)   # Neon cyan: all equal, unevaluated agents
COLOR_TRAIL_UNIFORM = (0,   90, 140)   # Dim teal (SRCALPHA base colour)
COLOR_TEXT_PRIMARY = (220, 235, 255)   # Bright white-blue HUD text
COLOR_TEXT_DIM     = (100, 125, 160)   # Secondary HUD text
COLOR_HUD_BG       = (8,   12,  25)    # HUD panel background (drawn with alpha)

# ---------------------------------------------------------------------------
# Trail Alpha Values  (used when drawing on a SRCALPHA surface)
# ---------------------------------------------------------------------------
TRAIL_ALPHA_ALIVE   = 55    # Semi-transparent cyan trails for active agents
TRAIL_ALPHA_BEST    = 140   # More opaque gold trail for the winner/best
TRAIL_ALPHA_UNIFORM = 45    # Very dim for the uniform active phase
