"""
environment.py — The simulation world.

Class hierarchy:
    BaseZone          Abstract coloured region on the canvas.
        StartZone     Spawn point; ants appear here each generation.
        TargetZone    Goal; entering this zone yields the highest fitness reward.
    Wall              Impassable rectangle; hitting one kills the ant immediately.
    Environment       Aggregates all zones and walls; exposes collision queries
                      and drives all rendering for the static scene.
"""

import math
import pygame
import config


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

class BaseZone:
    """
    Abstract base class for any named, coloured region on the simulation canvas.
    Concrete subclasses implement draw() to give each zone its distinct look.
    """

    def __init__(self, x: int, y: int, color: tuple):
        self._x     = x
        self._y     = y
        self._color = color

    @property
    def position(self) -> tuple:
        """Returns the (x, y) centre of the zone."""
        return (self._x, self._y)

    def draw(self, surface: pygame.Surface):
        """Render this zone onto the given surface. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement draw().")


class StartZone(BaseZone):
    """
    The spawn region for all ants at the beginning of each generation.
    Drawn as a filled circle to visually distinguish it from the target.
    """

    def __init__(self):
        super().__init__(config.START_X, config.START_Y, config.COLOR_START_ZONE)
        self._radius = 18

    def draw(self, surface: pygame.Surface):
        pygame.draw.circle(surface, self._color, (self._x, self._y), self._radius)
        # Inner white ring for clarity
        pygame.draw.circle(surface, (200, 255, 200), (self._x, self._y),
                           self._radius, 2)


class TargetZone(BaseZone):
    """
    The goal region. An ant whose centre enters this circle is considered
    successful and receives the maximum fitness bonus.
    A simple pulse animation draws visitors' eyes to the target.
    """

    def __init__(self):
        super().__init__(config.TARGET_X, config.TARGET_Y, config.COLOR_TARGET_ZONE)
        self._radius       = config.TARGET_RADIUS
        self._pulse_timer  = 0   # Counts frames for the oscillating visual effect

    @property
    def radius(self) -> int:
        return self._radius

    def contains(self, point: tuple) -> bool:
        """
        Returns True if (x, y) is inside the target circle.
        Uses squared-distance comparison to avoid a sqrt() call per frame.
        """
        dx = point[0] - self._x
        dy = point[1] - self._y
        return (dx * dx + dy * dy) <= (self._radius * self._radius)

    def update(self):
        """Advance the pulse animation counter by one simulation tick."""
        self._pulse_timer = (self._pulse_timer + 1) % 60

    def draw(self, surface: pygame.Surface):
        # Compute a ±4 px oscillation based on a triangle wave over 60 frames
        pulse = int(4 * abs(self._pulse_timer / 30.0 - 1.0))
        pygame.draw.circle(surface, self._color,
                           (self._x, self._y), self._radius + pulse)
        # Bright inner highlight ring
        pygame.draw.circle(surface, (255, 240, 160),
                           (self._x, self._y),
                           max(4, self._radius - 10 + pulse), 2)


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

class Wall:
    """
    An impassable rectangular obstacle.
    Ants that collide with a wall die immediately, receiving no further fitness
    reward — this strong negative pressure drives the EA to find wall-avoiding genomes.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self._rect = pygame.Rect(x, y, width, height)

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def collides_with_circle(self, center: tuple, radius: float) -> bool:
        """
        Circle-vs-AABB collision test using the nearest-point method.
        Finds the closest point on the rectangle to the circle centre and
        checks whether that distance is less than the radius.
        This is more accurate than a simple bounding-box test and avoids
        false positives when the ant brushes a wall corner.
        """
        cx, cy   = center
        closest_x = max(self._rect.left,  min(cx, self._rect.right))
        closest_y = max(self._rect.top,   min(cy, self._rect.bottom))
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) < (radius * radius)

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, config.COLOR_WALL, self._rect)
        # Top-edge highlight gives a subtle 3-D depth impression
        pygame.draw.line(surface, config.COLOR_WALL_EDGE,
                         self._rect.topleft, self._rect.topright, 2)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Aggregates the complete simulation world: boundary walls, internal maze
    walls, the start zone, and the target zone.

    Responsibilities:
        - Construct wall and zone objects from the config layout tables.
        - Answer collision queries so the Ant class can detect wall impacts.
        - Drive rendering of the static scene each frame.
    """

    def __init__(self):
        self._walls       = [Wall(x, y, w, h) for x, y, w, h in config.MAZE_WALLS]
        self._start_zone  = StartZone()
        self._target_zone = TargetZone()

    # --- Public accessors ---------------------------------------------------

    @property
    def start_position(self) -> tuple:
        return self._start_zone.position

    @property
    def target_zone(self) -> TargetZone:
        return self._target_zone

    # --- Collision API ------------------------------------------------------

    def is_colliding(self, position: tuple, radius: float) -> bool:
        """
        Returns True if a circular agent at 'position' with 'radius' overlaps
        any wall or lies outside the window boundaries.

        Called once per living ant per frame, so this must stay fast.
        The screen-boundary check uses simple comparisons; the wall check
        delegates to Wall.collides_with_circle() for accuracy.
        """
        cx, cy = position

        # Screen-edge boundary (treat window edges as instant-kill walls)
        if cx - radius < 0 or cx + radius > config.WINDOW_WIDTH:
            return True
        if cy - radius < 0 or cy + radius > config.WINDOW_HEIGHT:
            return True

        # Internal maze walls
        for wall in self._walls:
            if wall.collides_with_circle(position, radius):
                return True

        return False

    # --- Update & Rendering -------------------------------------------------

    def update(self):
        """Advance animated elements (currently just the target zone pulse)."""
        self._target_zone.update()

    def draw(self, surface: pygame.Surface):
        """Render the environment in correct depth order: zones behind walls."""
        # Zones first so walls visually sit on top
        self._start_zone.draw(surface)
        self._target_zone.draw(surface)

        # All maze and boundary walls
        for wall in self._walls:
            wall.draw(surface)
