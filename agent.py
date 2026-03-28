"""
agent.py — Virtual agent definitions for the Smart Swarm simulation.

Class hierarchy:
    BaseAgent   Abstract base; defines the interface every agent must satisfy.
        Ant     Concrete ant agent navigated step-by-step by its DNA genome.

Genotype / Phenotype mapping
-----------------------------
Genotype  : a list of DNA_LENGTH floats, each in [-1, 1].
            Each gene encodes the *heading change* for one movement step.

Phenotype : the ant turns by (gene × ANT_MAX_ANGLE_DELTA) degrees, then
            moves forward ANT_SPEED pixels in the new direction.
            This produces curved, organic-looking trajectories whose shape
            is entirely determined by the gene sequence — the classic
            genotype-to-phenotype mapping studied in evolutionary biology.
"""

import math
import pygame
import config


class BaseAgent:
    """
    Abstract base class for all simulation agents.
    Enforces a consistent interface (update / draw / compute_fitness) so the
    simulation engine can treat any agent type polymorphically.
    """

    def __init__(self, position: tuple, genome: list):
        # _position is mutable [x, y] so it can be updated in-place each step
        self._position = list(position)
        self._genome   = genome
        self._is_alive = True
        self._fitness  = 0.0

    # --- Read-only properties -----------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def fitness(self) -> float:
        return self._fitness

    # --- Interface methods (must be overridden) ------------------------------

    def update(self, environment):
        """Advance the agent by one simulation step."""
        raise NotImplementedError

    def draw(self, surface: pygame.Surface, is_best: bool = False):
        """Render the agent onto the given surface."""
        raise NotImplementedError

    def compute_fitness(self, target_position: tuple,
                        max_possible_distance: float) -> float:
        """Evaluate and store the agent's fitness score."""
        raise NotImplementedError


class Ant(BaseAgent):
    """
    A virtual ant that reads its genome one gene at a time to navigate the maze.

    Movement model
    --------------
    Each tick the ant:
        1. Reads gene[step]  (a float ∈ [-1, 1]).
        2. Adjusts its current heading by gene × ANT_MAX_ANGLE_DELTA degrees.
        3. Moves ANT_SPEED pixels in the new heading direction.
        4. Dies if the new position collides with a wall or screen boundary.
        5. Stops (marks reached_target = True) if it enters the target zone.

    Fitness function
    ----------------
    Reaching the target  → 10 000 bonus + steps survived (speed reward).
    Not reaching target  → proximity score ∈ [0, 1 000] based on how close
                           the ant ever got, plus a small longevity bonus.
                           This creates a smooth gradient that guides the EA
                           even when no ant reaches the goal yet.
    """

    def __init__(self, position: tuple, genome: list, index: int):
        super().__init__(position, genome)
        self._index           = index        # Matches the PyGAD solution index
        self._heading         = 0.0          # Current direction in degrees (0 = east)
        self._step            = 0            # Which gene we are currently executing
        self._steps_survived  = 0            # How many steps completed before death
        self._reached_target  = False

        # _has_won is the global-stop signal: set to True the instant this ant's
        # position enters the target zone. SimulationEngine polls this flag each
        # frame to decide whether to halt the entire evolutionary run.
        self._has_won         = False

        # _trail stores (x, y) positions for drawing the movement path.
        # Capped at DNA_LENGTH so the winner's complete path stays fully visible
        # on screen — a trail capped at 50 would hide most of the winning route.
        self._trail           = [tuple(position)]
        self._trail_max_len   = config.DNA_LENGTH

        # Track the closest the ant ever got to the target (Euclidean).
        # Used as a secondary fitness signal for the final approach to target.
        self._closest_distance = float('inf')

        # Track the furthest-right (max x) position ever reached.
        # This is a one-way ratchet: looping brings the ant back left, so loops
        # do NOT increase _max_x_reached. This makes it loop-proof as a primary
        # fitness driver, unlike _closest_distance which can be gamed by briefly
        # swinging near the target then looping away for hundreds of steps.
        self._max_x_reached = self._position[0]

    # --- Properties ---------------------------------------------------------

    @property
    def index(self) -> int:
        return self._index

    @property
    def reached_target(self) -> bool:
        return self._reached_target

    @property
    def has_won(self) -> bool:
        """
        True the moment this ant entered the target zone.
        SimulationEngine reads this flag each frame to trigger the global
        win state — halting evolution and showing the victory overlay.
        """
        return self._has_won

    @property
    def steps_survived(self) -> int:
        return self._steps_survived

    @property
    def position(self) -> tuple:
        return (self._position[0], self._position[1])

    # --- Core simulation methods --------------------------------------------

    def update(self, environment):
        """
        Execute one gene from the DNA sequence.

        Steps:
            1a. Guard — skip if already dead (wall collision or prior target hit).
            1b. Guard — genome exhausted: perform a final proximity check at the
                        ant's current (last-committed) position before marking dead.
                        This ensures compute_fitness() scores the true closest
                        approach, not a stale value from an earlier step.
            2.  Decode gene → heading delta.
            3.  Compute new (x, y) from current heading and speed.
            4.  Collision check — die on wall/boundary impact.
            5.  Commit move, update trail and survival counters.
            6.  Target check — uses true 2D Euclidean distance sqrt(dx²+dy²);
                no 1D shortcut. Stop if the circle centred on the target contains
                the ant's new position.
        """
        # Guard 1a — ant was already killed by a wall or previous target detection
        if not self._is_alive:
            return

        # Guard 1b — genome exhausted: all DNA instructions have been consumed.
        # Before dying, run a final distance measurement at the ant's current
        # resting position. This is necessary because the previous step's
        # _closest_distance update used the position AFTER that step's movement;
        # we make it explicit here that the final committed position is authoritative.
        if self._step >= len(self._genome):
            target_x, target_y = environment.target_zone.position
            dx = self._position[0] - target_x
            dy = self._position[1] - target_y
            # True 2D Euclidean distance — NOT a 1D axis comparison
            final_dist = math.sqrt(dx * dx + dy * dy)
            if final_dist < self._closest_distance:
                self._closest_distance = final_dist
            # Defensive: if the final resting position is inside the target zone
            # (e.g. entered on the very last gene but check below hadn't fired),
            # award the win and raise the global-stop signal before marking dead.
            if final_dist <= environment.target_zone.radius:
                self._reached_target = True
                self._has_won        = True
            self._is_alive = False
            return

        # Step 2 — Decode gene: linear map [-1,1] → [-max_delta, +max_delta] degrees.
        # Using a linear map keeps the genome space interpretable and makes crossover
        # offspring behaviourally coherent.
        gene = self._genome[self._step]
        self._heading += gene * config.ANT_MAX_ANGLE_DELTA
        self._step += 1

        # Step 3 — Compute displacement vector from heading angle
        rad = math.radians(self._heading)
        new_x = self._position[0] + math.cos(rad) * config.ANT_SPEED
        new_y = self._position[1] + math.sin(rad) * config.ANT_SPEED

        # Step 4 — Collision detection: die if the new position overlaps any wall
        # or strays outside the window boundary.
        if environment.is_colliding((new_x, new_y), config.ANT_RADIUS):
            self._is_alive = False
            return   # Do NOT commit; ant's position freezes at last valid spot

        # Step 5 — Commit the valid move
        self._position[0] = new_x
        self._position[1] = new_y
        self._steps_survived += 1

        # Advance the rightward-progress ratchet — never decreases, so loops
        # that bring the ant back left earn zero additional fitness credit.
        if new_x > self._max_x_reached:
            self._max_x_reached = new_x

        # Update rolling trail buffer (FIFO, capped to _trail_max_len)
        self._trail.append((new_x, new_y))
        if len(self._trail) > self._trail_max_len:
            self._trail.pop(0)

        # Step 6 — Track closest approach and check win condition.
        # IMPORTANT: distance is always the true 2D Euclidean distance using
        # both axes. A 1D check (e.g. new_y == target_y) would falsely trigger
        # when the ant is vertically aligned but not yet horizontally at the target.
        target_x, target_y = environment.target_zone.position
        dx = new_x - target_x
        dy = new_y - target_y
        dist = math.sqrt(dx * dx + dy * dy)   # Full 2D distance, no axis shortcut

        if dist < self._closest_distance:
            self._closest_distance = dist

        if dist <= environment.target_zone.radius:
            # Set both flags atomically: _reached_target feeds compute_fitness();
            # _has_won is the global-stop signal polled by SimulationEngine each frame.
            self._reached_target = True
            self._has_won        = True
            self._is_alive       = False  # Stop this ant; engine detects _has_won next frame

    def compute_fitness(self, target_position: tuple,
                        max_possible_distance: float) -> float:
        """
        Evaluate this ant's performance and cache the result in self._fitness.

        Fitness landscape design — why the old formula failed and what changed:

        OLD formula:  proximity_ratio * 1000  +  steps_survived * 0.3
          Problem:  steps_survived could add up to 240 pts (800 steps × 0.3).
                    _closest_distance records the LIFETIME minimum, so an ant
                    that briefly swings near the target on step 50 of a 800-step
                    loop keeps that proximity score forever. Looping was actively
                    rewarded because it maximised both terms simultaneously.
                    The "best" ant therefore became the one that loops most
                    efficiently, not the one that progresses furthest.

        NEW formula:
          1. X-progress (primary, 0–600 pts):
             Based on _max_x_reached, which is a one-way ratchet — looping
             never increases it. This directly rewards moving right toward the
             target without being gameable by circular paths.

          2. Wall-clearing milestones (+75 / +150 / +300 pts):
             Discrete score cliffs at each wall's right edge. A genome that
             learns to navigate PAST wall C scores 300 pts more than one that
             hits its face — creating unmistakable selection pressure for the
             exact manoeuvre needed (dip south before x=650).

          3. Euclidean proximity bonus (0–400 pts, quadratic):
             Squared proximity ratio pulls the gradient toward the target once
             the ant is already far right. Quadratic weighting means the bonus
             only becomes significant when the ant is VERY close, preventing an
             ant that flew past the target zone in a loop from gaming this term.

          4. Longevity (0–40 pts max):
             Reduced from 0.3/step (max 240) to 0.05/step (max 40). Still
             prevents genomes that kamikaze into wall A from tying with genomes
             that at least survive a few steps, but too small to incentivise
             staying alive via looping.
        """
        if self._reached_target:
            # Massive flat bonus ensures any winner dominates every non-winner
            # regardless of how many steps it took.
            self._fitness = 10_000.0 + self._steps_survived
            return self._fitness

        # --- Component 1: x-progress (loop-proof forward progress) -----------
        x_span        = config.TARGET_X - config.START_X          # 780 px
        x_progress    = max(0.0, self._max_x_reached - config.START_X)
        x_ratio       = min(1.0, x_progress / x_span)             # clamp [0,1]
        score         = x_ratio * 600.0

        # --- Component 2: wall-clearing milestones ---------------------------
        # Each milestone fires only once (highest cleared wall wins).
        # The gaps are: under A (y>335), over B (y<265), under C (y>335).
        # An ant that hits wall C head-on never reaches WALL_C_CLEAR_X,
        # so it gets 0 pts here even if it loops for 800 steps nearby.
        if self._max_x_reached >= config.WALL_C_CLEAR_X:
            score += 300.0   # Cleared all 3 walls — within reach of target
        elif self._max_x_reached >= config.WALL_B_CLEAR_X:
            score += 150.0   # Cleared walls A and B
        elif self._max_x_reached >= config.WALL_A_CLEAR_X:
            score += 75.0    # Cleared wall A only

        # --- Component 3: Euclidean proximity (final-approach pull) ----------
        # Squared ratio heavily discounts ants that are merely "somewhat close";
        # the bonus only grows large when the ant is genuinely near the target.
        if self._closest_distance < float('inf'):
            prox_ratio = max(0.0, (max_possible_distance - self._closest_distance)
                                  / max_possible_distance)
            score += (prox_ratio ** 2) * 400.0

        # --- Component 4: tiny longevity (anti-kamikaze only) ----------------
        score += self._steps_survived * 0.05   # max 40 pts for 800 steps

        self._fitness = score
        return self._fitness

    # --- Rendering ----------------------------------------------------------

    def draw(self, surface: pygame.Surface, is_best: bool = False,
             active_uniform: bool = False, trail_width: int = 1):
        """
        Render this ant and its movement trail.

        Parameters
        ----------
        is_best        : True during the evaluation pause or victory screen to
                         draw this ant in gold with its full trail visible.
        active_uniform : True during the live simulation phase. All agents —
                         alive AND dead — are rendered in the same neutral colour
                         so no ant looks like a "predicted winner" before fitness
                         has been computed. Trails are shown for every agent.
        trail_width    : Line width for the trail polyline. Pass 3 during the
                         evaluation pause so the winner's path is clearly thick.
        """
        if not self._trail:
            return

        # --- Colour selection --------------------------------------------------
        # Priority order:
        #   1. active_uniform  → all ants identical (fitness unknown)
        #   2. is_best / won   → gold (fitness confirmed, evaluation phase)
        #   3. is_alive        → standard blue (normal rendering)
        #   4. dead            → dim red, no trail (normal rendering)
        if active_uniform:
            # During the active generation, fitness is not yet known.
            # Every ant — alive or crashed — uses the same neutral colour so
            # the audience cannot be misled by a "predicted" frontrunner.
            body_color  = config.COLOR_ANT_UNIFORM
            trail_color = config.COLOR_TRAIL_UNIFORM
        elif is_best or self._has_won:
            # Post-generation evaluation or victory screen: confirmed best/winner.
            body_color  = config.COLOR_ANT_BEST
            trail_color = config.COLOR_TRAIL_BEST
        elif self._is_alive:
            body_color  = config.COLOR_ANT_ALIVE
            trail_color = config.COLOR_TRAIL_ALIVE
        else:
            body_color  = config.COLOR_ANT_DEAD
            trail_color = None   # Dead non-winners: body dot only, no trail

        # --- Trail polyline ----------------------------------------------------
        if trail_color and len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                pygame.draw.line(
                    surface, trail_color,
                    (int(self._trail[i - 1][0]), int(self._trail[i - 1][1])),
                    (int(self._trail[i][0]),     int(self._trail[i][1])),
                    trail_width,   # 1 px during active phase; 3 px for evaluation winner
                )

        # --- Body circle -------------------------------------------------------
        cx = int(self._position[0])
        cy = int(self._position[1])
        pygame.draw.circle(surface, body_color, (cx, cy), config.ANT_RADIUS)

        # --- Direction indicator -----------------------------------------------
        # Show the heading arrow for living ants, the confirmed best/winner, and
        # ants drawn in the uniform active phase (all are still navigating).
        if self._is_alive or is_best or self._has_won or active_uniform:
            rad   = math.radians(self._heading)
            tip_x = cx + int(math.cos(rad) * (config.ANT_RADIUS + 4))
            tip_y = cy + int(math.sin(rad) * (config.ANT_RADIUS + 4))
            pygame.draw.line(surface, (255, 255, 255), (cx, cy), (tip_x, tip_y), 2)
