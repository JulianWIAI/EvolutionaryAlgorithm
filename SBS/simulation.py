"""
simulation.py — Orchestrates the full evolutionary loop.

Architecture overview
---------------------
The SimulationEngine owns:
    - A Pygame window and rendering pipeline.
    - An Environment (maze, zones).
    - A list of Ant agents representing the current generation.
    - A _GeneticAlgorithm instance managing selection, crossover, mutation.

Application State Machine  (AppState enum)
------------------------------------------
START_SCREEN  Welcome/title screen; displays exhibition highscore.
SIMULATING    One generation is running; ants navigate the maze.
EVALUATING    Post-generation pause (1.5 s); best ant highlighted in gold.
SUCCESS       A winner reached the target; 5-second victory banner.
RESET_LOOP    Banner shifts to "Click or Touch Screen to Reset" prompt.

Transitions:
    START_SCREEN → SIMULATING   on click / touch
    SIMULATING   → EVALUATING   all ants done, no winner this generation
    SIMULATING   → SUCCESS      a winning ant detected
    EVALUATING   → SIMULATING   after 1.5 s evaluation pause
    SUCCESS      → RESET_LOOP   after 5 s auto-timer
    SUCCESS      → START_SCREEN on click / touch  (full reset)
    RESET_LOOP   → START_SCREEN on click / touch  (full reset)

Exhibition highscore
--------------------
_exhibition_highscore persists across all resets within one process run.
It is displayed on the START_SCREEN and in the HUD during SIMULATING so
museum visitors can see whether a new run beat the all-time record.

Pygbag / asyncio note
---------------------
run() is declared async and ends every frame with `await asyncio.sleep(0)`
to yield control to the browser event loop — the required Pygbag pattern.
"""

import asyncio
import enum
import math
import os
import random
import pygame

from . import config
from .environment import Environment
from .agent import Ant


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState(enum.Enum):
    """
    Discrete states driving the exhibition loop.
    See module docstring for full transition diagram.
    """
    START_SCREEN = "start_screen"
    SIMULATING   = "simulating"
    EVALUATING   = "evaluating"
    SUCCESS      = "success"
    RESET_LOOP   = "reset_loop"


# ---------------------------------------------------------------------------
# Pure-Python Genetic Algorithm (Pygbag-compatible, no numpy)
# ---------------------------------------------------------------------------

class _GeneticAlgorithm:
    """
    Minimal genetic algorithm using only the standard-library random module.

    Replicates the PyGAD configuration that was previously used:
        - Rank-based parent selection (top num_parents_mating survive)
        - Single-point crossover between randomly chosen parent pairs
        - Random gene-replacement mutation on non-elite offspring
        - Elitism: the best keep_parents solutions are copied unchanged

    The population is stored as a list-of-lists so it requires no
    third-party packages and works identically on CPython and Pyodide/WASM.
    """

    def __init__(self, population_size: int, num_genes: int,
                 gene_low: float, gene_high: float,
                 num_parents_mating: int, mutation_percent_genes: float,
                 keep_parents: int = 2):
        self._pop_size    = population_size
        self._num_genes   = num_genes
        self._gene_low    = gene_low
        self._gene_high   = gene_high
        self._num_parents = num_parents_mating
        self._mut_pct     = mutation_percent_genes
        self._keep        = keep_parents

        # Initialise a random starting population as a list of lists.
        # Each inner list is one genome of length num_genes with values in
        # [gene_low, gene_high].
        self.population = [
            [random.uniform(gene_low, gene_high) for _ in range(num_genes)]
            for _ in range(population_size)
        ]

    def evolve(self, fitness_scores: list) -> list:
        """
        Produce the next generation from the current population.

        Parameters
        ----------
        fitness_scores : list[float]
            One fitness value per row in self.population, in index order
            (score for ant 0 → fitness_scores[0], etc.).

        Returns
        -------
        list  The new population (also stored in self.population).
        """
        n   = self._pop_size
        pop = self.population

        # --- 1. Rank-based selection: choose the top num_parents_mating -------
        ranked  = sorted(range(len(fitness_scores)),
                         key=lambda i: fitness_scores[i], reverse=True)
        parents = [pop[ranked[i]] for i in range(self._num_parents)]

        # --- 2. Elitism: carry the best keep_parents solutions unchanged ------
        new_pop = [list(parents[i]) for i in range(self._keep)]

        # --- 3. Single-point crossover to fill the remainder -----------------
        needed = n - self._keep
        for _ in range(needed):
            a = random.randrange(self._num_parents)
            b = random.randrange(self._num_parents)
            # Guarantee the two parents are always distinct.
            if a == b:
                b = (b + 1) % self._num_parents
            cut   = random.randint(1, self._num_genes - 1)
            child = parents[a][:cut] + parents[b][cut:]
            new_pop.append(child)

        new_pop = new_pop[:n]

        # --- 4. Mutate non-elite offspring -----------------------------------
        num_mutations = max(1, int(self._num_genes * self._mut_pct / 100))
        for i in range(self._keep, n):
            loci = random.sample(range(self._num_genes), num_mutations)
            for loc in loci:
                new_pop[i][loc] = random.uniform(self._gene_low, self._gene_high)

        self.population = new_pop
        return self.population


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Master controller for the Smart Swarm simulation.

    Private attributes (single-underscore = internal state):
        _screen               Pygame display surface.
        _clock                Pygame clock for frame-rate control.
        _font_title           Large display font for the start screen title.
        _font_large           Medium bold font for banners and HUD headers.
        _font_small           Small font for HUD body text.
        _trail_surface        SRCALPHA surface for semi-transparent ant trails.
        _exhibition_highscore All-time best fitness across every run this session.
        _environment          The maze / zone / wall scene.
        _ants                 Active ant population for the current generation.
        _generation           Counter incremented each generational cycle.
        _best_fitness_ever    Highest fitness score in this run.
        _best_fitness_gen     Highest fitness score in the current generation.
        _max_distance         Straight-line start-to-target distance (normalisation).
        _cached_fitness       Dict[ant_index → fitness] populated each generation.
        _ga                   _GeneticAlgorithm instance managing the population.
        _lucky_few            Elite parents carried over unchanged (elitism count).
        _chance_of_mutation   Percentage of genes mutated each generation.
        _stagnation_count     Consecutive generations with negligible fitness gain.
        _prev_best_fitness    Best fitness at the last generation where gain > 5 pts.
    """

    def __init__(self):
        pygame.init()

        # set_icon() MUST be called before set_mode() — SDL2 on macOS ignores
        # icons set after the window is created (Dock icon stays as Python default).
        # The raw icon.png can be any size; we scale to 256×256 which SDL2 uses
        # for the macOS Dock and Windows taskbar at full quality.
        try:
            _icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets", "icon.png"
            )
            _icon_raw  = pygame.image.load(_icon_path)
            _icon_surf = pygame.transform.smoothscale(_icon_raw, (256, 256))
            pygame.display.set_icon(_icon_surf)
        except Exception:
            pass

        pygame.display.set_caption(config.WINDOW_TITLE)
        self._screen = pygame.display.set_mode((config.WINDOW_WIDTH,
                                                config.WINDOW_HEIGHT))

        self._clock  = pygame.time.Clock()

        # Prefer a clean modern sans-serif; SysFont accepts a comma-separated
        # priority list and picks the first family that exists on the system.
        _sans = "Arial, Segoe UI, Roboto, Helvetica, DejaVu Sans"
        self._font_title = pygame.font.SysFont(_sans, 46, bold=True)
        self._font_large = pygame.font.SysFont(_sans, 26, bold=True)
        self._font_small = pygame.font.SysFont(_sans, 15)

        # Fallback to the built-in default font if none of the above are found
        # (SysFont returns None on some WASM builds when the family is missing)
        if self._font_title is None:
            self._font_title = pygame.font.Font(None, 56)
        if self._font_large is None:
            self._font_large = pygame.font.Font(None, 32)
        if self._font_small is None:
            self._font_small = pygame.font.Font(None, 20)

        # Dedicated SRCALPHA surface for semi-transparent agent trails.
        # Cleared each frame, drawn on by every ant, then composited onto the
        # main screen — keeping the dark background from being polluted.
        self._trail_surface = pygame.Surface(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA
        )

        # Exhibition highscore: fewest generations needed to reach the target.
        # Starts at 999 (sentinel = "no record yet"); only updated on SUCCESS.
        # Persists across all resets for the session.
        self._exhibition_highscore = 999

        # Constant EA hyper-parameters (never reset between runs)
        self._lucky_few          = config.NUM_PARENTS_MATING
        self._chance_of_mutation = config.MUTATION_PERCENT_GENES

        # Initialise all per-run state for the very first run
        self._init_run_state()

    # -----------------------------------------------------------------------
    # Per-Run Initialisation / Reset
    # -----------------------------------------------------------------------

    def _init_run_state(self):
        """
        (Re-)initialise all per-run variables.

        Called once at startup and again every time the visitor resets the
        exhibition.  _exhibition_highscore and the font/display objects are
        deliberately excluded — they persist for the entire process lifetime.
        """
        self._environment       = Environment()
        self._ants              = []
        self._generation        = 0
        self._best_fitness_ever = 0.0
        self._best_fitness_gen  = 0.0
        self._stagnation_count  = 0
        self._prev_best_fitness = 0.0
        self._cached_fitness    = {}

        # Pre-compute straight-line start → target distance for fitness normalisation
        sx, sy = self._environment.start_position
        tx, ty = self._environment.target_zone.position
        self._max_distance = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)

        # Fresh randomly-seeded GA instance — new run, clean slate
        self._ga = _GeneticAlgorithm(
            population_size        = config.POPULATION_SIZE,
            num_genes              = config.DNA_LENGTH,
            gene_low               = -1.0,
            gene_high              =  1.0,
            num_parents_mating     = self._lucky_few,
            mutation_percent_genes = self._chance_of_mutation,
            keep_parents           = 2,
        )

    # -----------------------------------------------------------------------
    # Population Evolution
    # -----------------------------------------------------------------------

    def _evolve_population(self) -> list:
        """
        Trigger one round of EA operators (selection → crossover → mutation).

        Fitness scores are read from _cached_fitness and passed directly to
        _GeneticAlgorithm.evolve(), which returns the offspring population.
        """
        n            = len(self._ga.population)
        fitness_list = [self._cached_fitness.get(i, 0.0) for i in range(n)]
        return self._ga.evolve(fitness_list)

    def _check_stagnation_and_inject(self, population: list) -> list:
        """
        Detect fitness stagnation and restore diversity when the EA is stuck.

        If the generation best has not improved by more than 5 fitness points
        for 8 consecutive generations, 30 % of the population (the weakest
        individuals) are replaced with fresh random genomes.  The counter then
        resets so the EA gets another 8 generations to explore before the next
        injection.

        The 5-point / 8-generation thresholds are tuned for the slalom maze:
          • Wall milestones award 75–300 pts.  A 5-pt threshold avoids false
            resets from tiny proximity improvements while still detecting genuine
            stagnation at each wall barrier.
          • 8 generations gives the EA enough time to consolidate a new milestone
            before concluding that it has converged on a local optimum.
        """
        if self._best_fitness_gen > self._prev_best_fitness + 5.0:
            # Meaningful improvement this generation — reset the counter.
            self._stagnation_count  = 0
            self._prev_best_fitness = self._best_fitness_gen
            return population

        self._stagnation_count += 1
        if self._stagnation_count < 8:
            return population

        # --- Inject fresh blood -----------------------------------------------
        n        = len(population)
        n_inject = max(2, n * 3 // 10)   # 30 %, minimum 2

        # Sort population best-first so we keep the good genomes and replace
        # only the weakest individuals.
        order      = sorted(range(n),
                            key=lambda i: self._cached_fitness.get(i, 0.0),
                            reverse=True)
        sorted_pop = [list(population[order[i]]) for i in range(n)]
        for i in range(n - n_inject, n):
            sorted_pop[i] = [random.uniform(-1.0, 1.0)
                             for _ in range(config.DNA_LENGTH)]

        # Sync the GA's internal population reference so the next evolve() call
        # operates on the injected population, not the pre-injection one.
        self._ga.population    = sorted_pop
        self._stagnation_count = 0
        return sorted_pop

    # -----------------------------------------------------------------------
    # Generation Management
    # -----------------------------------------------------------------------

    def _spawn_generation(self, population: list):
        """
        Instantiate one Ant per row in the population list.
        All ants start at the environment's designated start position with
        their index matching the population row index (0 … N-1).
        """
        start_pos  = self._environment.start_position
        self._ants = [
            Ant(position=start_pos,
                genome=population[i],
                index=i)
            for i in range(len(population))
        ]

    def _all_ants_done(self) -> bool:
        """
        Returns True once every ant has either crashed or exhausted its genome.
        This is the termination condition for the per-generation simulation.
        """
        return all(not ant.is_alive for ant in self._ants)

    def _compute_and_cache_fitness(self):
        """
        Call compute_fitness() on every ant and store results in _cached_fitness.
        Also update the generation and all-time best fitness trackers for the HUD.
        """
        self._cached_fitness = {}
        target_pos = self._environment.target_zone.position

        for ant in self._ants:
            score = ant.compute_fitness(target_pos, self._max_distance)
            self._cached_fitness[ant.index] = score

        if self._cached_fitness:
            self._best_fitness_gen = max(self._cached_fitness.values())
            if self._best_fitness_gen > self._best_fitness_ever:
                self._best_fitness_ever = self._best_fitness_gen

    def _get_best_ant_index(self) -> int:
        """Return the population index of the highest-scoring ant this generation."""
        if not self._cached_fitness:
            return 0
        return max(self._cached_fitness, key=lambda i: self._cached_fitness[i])

    def _generation_has_winner(self) -> bool:
        """
        Returns True if any ant in the current generation has set _has_won.
        Called once per frame inside the simulation state so the engine
        can halt the moment a winner is detected.
        """
        return any(ant.has_won for ant in self._ants)

    def _find_winner_index(self) -> int:
        """
        Returns the population index of the first ant that reached the target,
        or -1 if no winner exists.
        """
        for ant in self._ants:
            if ant.has_won:
                return ant.index
        return -1

    # -----------------------------------------------------------------------
    # Rendering Helpers
    # -----------------------------------------------------------------------

    def _blit_text_shadowed(self, surface: pygame.Surface,
                            font: pygame.font.Font, text: str,
                            color: tuple, pos: tuple,
                            shadow_offset: tuple = (2, 2)):
        """
        Blit text with a dark drop-shadow so it pops against any background.
        The shadow is rendered first at pos+shadow_offset, then the coloured
        glyph is drawn on top at pos.
        """
        shadow_surf = font.render(text, True, (0, 0, 0))
        glyph_surf  = font.render(text, True, color)
        surface.blit(shadow_surf, (pos[0] + shadow_offset[0],
                                   pos[1] + shadow_offset[1]))
        surface.blit(glyph_surf, pos)

    def _blit_text_glowing(self, surface: pygame.Surface,
                           font: pygame.font.Font, text: str,
                           color: tuple, glow_color: tuple, pos: tuple):
        """
        Render text with a multi-directional colour halo for a neon glow effect.
        The glow is drawn at ±2 px offsets in glow_color, then the main glyph
        is composited on top in color.  Used on the start screen title.
        """
        glow_surf = font.render(text, True, glow_color)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                surface.blit(glow_surf, (pos[0] + dx, pos[1] + dy))
        main_surf = font.render(text, True, color)
        surface.blit(main_surf, pos)

    # -----------------------------------------------------------------------
    # Scene Rendering
    # -----------------------------------------------------------------------

    def _draw_hud(self, alive_count: int):
        """
        Render the heads-up display panel in the top-left corner.

        Shows: exhibition highscore, generation number, all-time and current-gen
        best fitness, and number of still-living ants so visitors can track
        evolutionary progress across and within each run.
        """
        _hs_display = ("None" if self._exhibition_highscore == 999
                       else f"{self._exhibition_highscore} Gens")
        lines = [
            ("Exhibition HS", _hs_display),
            ("Generation",    str(self._generation)),
            ("Best (ever)",   f"{self._best_fitness_ever:,.1f}"),
            ("Best (gen)",    f"{self._best_fitness_gen:,.1f}"),
            ("Alive",         f"{alive_count} / {len(self._ants)}"),
        ]

        padding     = 10
        line_height = 22
        panel_w     = 280
        panel_h     = len(lines) * line_height + 2 * padding

        hud_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        hud_surf.fill((*config.COLOR_HUD_BG, 190))
        self._screen.blit(hud_surf, (8, 8))

        for i, (label, value) in enumerate(lines):
            label_text = f"{label}: "
            label_surf = self._font_small.render(label_text, True,
                                                 config.COLOR_TEXT_DIM)
            # Exhibition HS line rendered in gold to make it stand out
            value_color = (config.COLOR_ANT_BEST if i == 0
                           else config.COLOR_TEXT_PRIMARY)
            y_pos  = 8 + padding + i * line_height
            x_base = 8 + padding
            self._blit_text_shadowed(self._screen, self._font_small,
                                     label_text, config.COLOR_TEXT_DIM,
                                     (x_base, y_pos))
            self._blit_text_shadowed(self._screen, self._font_small,
                                     value, value_color,
                                     (x_base + label_surf.get_width(), y_pos))

    def _draw_zone_labels(self):
        """Render 'S' and 'T' text labels centred on the start and target zones."""
        sx, sy = self._environment.start_position
        tx, ty = self._environment.target_zone.position
        for text, x, y in [("S", sx, sy), ("T", tx, ty)]:
            surf   = self._font_small.render(text, True, (255, 255, 255))
            blit_x = x - surf.get_width()  // 2
            blit_y = y - surf.get_height() // 2
            self._blit_text_shadowed(self._screen, self._font_small,
                                     text, (255, 255, 255), (blit_x, blit_y))

    def _draw_generation_banner(self):
        """
        Render a large centred generation counter at the top of the screen.
        This is the element showcase visitors notice first from a distance.
        """
        text  = f"Generation  {self._generation}"
        surf  = self._font_large.render(text, True, config.COLOR_TEXT_PRIMARY)
        x_pos = config.WINDOW_WIDTH // 2 - surf.get_width() // 2
        self._blit_text_shadowed(self._screen, self._font_large,
                                 text, config.COLOR_TEXT_PRIMARY, (x_pos, 6))

    def _draw_scene(self, best_ant_index: int, active_phase: bool = False):
        """
        Paint the full simulation scene onto self._screen WITHOUT flipping.

        active_phase=True  → generation is still running; fitness is unknown.
            Every ant is drawn with active_uniform=True so no agent appears
            to be a "predicted winner". All trails are visible in a neutral colour.

        active_phase=False → post-generation or victory screen; fitness is known.
            best_ant_index == -1 means no evaluated best exists yet (sentinel).
            Otherwise the best / winner is drawn in gold on top.
        """
        self._screen.fill(config.COLOR_BG)
        self._environment.draw(self._screen)

        # Clear the alpha trail surface; trails will be composited after bodies
        self._trail_surface.fill((0, 0, 0, 0))

        if active_phase:
            for ant in self._ants:
                ant.draw(self._screen, active_uniform=True,
                         trail_surface=self._trail_surface)
        else:
            has_known_best = (best_ant_index >= 0)

            # Dead non-best ants first (rendered behind living ants)
            for ant in self._ants:
                is_this_the_best = has_known_best and ant.index == best_ant_index
                if not ant.is_alive and not is_this_the_best:
                    ant.draw(self._screen, is_best=False,
                             trail_surface=self._trail_surface)

            # Living ants
            for ant in self._ants:
                if ant.is_alive:
                    ant.draw(self._screen,
                             is_best=(has_known_best and ant.index == best_ant_index),
                             trail_surface=self._trail_surface)

            # Confirmed best / winner drawn last so it sits on top of everything
            if has_known_best and self._ants:
                best_ant = self._ants[best_ant_index]
                if not best_ant.is_alive:
                    best_ant.draw(self._screen, is_best=True,
                                  trail_surface=self._trail_surface)

        # Composite semi-transparent trails over the scene
        self._screen.blit(self._trail_surface, (0, 0))

        self._draw_zone_labels()
        alive_count = sum(1 for a in self._ants if a.is_alive)
        self._draw_hud(alive_count)
        self._draw_generation_banner()

    def _render_frame(self, best_ant_index: int, active_phase: bool = False):
        """Draw the scene and flip the display — the normal per-frame call."""
        self._draw_scene(best_ant_index, active_phase=active_phase)
        pygame.display.flip()

    def _draw_evaluation_scene(self, best_ant_index: int):
        """
        Render the post-generation evaluation frame.

        All ants except the confirmed best are drawn as dim dead dots (no trail).
        The best ant is drawn in gold with a thick trail so visitors can trace
        the exact path the EA judged to be strongest this generation.
        """
        self._screen.fill(config.COLOR_BG)
        self._environment.draw(self._screen)

        self._trail_surface.fill((0, 0, 0, 0))
        has_known_best = (best_ant_index >= 0)

        for ant in self._ants:
            is_this_the_best = has_known_best and ant.index == best_ant_index
            if not is_this_the_best:
                ant.draw(self._screen, is_best=False,
                         trail_surface=self._trail_surface)

        if has_known_best and self._ants:
            best_ant = self._ants[best_ant_index]
            best_ant.draw(self._screen, is_best=True, trail_width=3,
                          trail_surface=self._trail_surface)

        # Composite semi-transparent trails — best ant trail (width=3, alpha=140)
        # is clearly legible against the dark background
        self._screen.blit(self._trail_surface, (0, 0))

        self._draw_zone_labels()
        alive_count = sum(1 for a in self._ants if a.is_alive)
        self._draw_hud(alive_count)
        self._draw_generation_banner()

    def _draw_win_overlay(self, winning_generation: int,
                          reset_mode: bool = False):
        """
        Composite a centred victory banner on top of whatever is already drawn
        on self._screen. Does NOT call pygame.display.flip() — the caller flips
        after this so the banner and scene appear in a single atomic update.

        reset_mode=False  SUCCESS state  — standard victory headline.
        reset_mode=True   RESET_LOOP     — subline prompts visitor to touch/click.
        """
        _hs_text = ("None" if self._exhibition_highscore == 999
                    else f"{self._exhibition_highscore} Gens")
        headline = f"Target Reached in Generation {winning_generation}!"
        subline  = ("Click or Touch Screen to Reset"
                    if reset_mode else
                    f"Exhibition Record: {_hs_text}")

        headline_surf = self._font_large.render(headline, True, (255, 240, 80))
        subline_surf  = self._font_small.render(subline,  True, (200, 200, 210))

        padding = 28
        panel_w = max(headline_surf.get_width(),
                      subline_surf.get_width()) + padding * 2
        panel_h = (headline_surf.get_height()
                   + subline_surf.get_height()
                   + padding * 3)
        panel_x = config.WINDOW_WIDTH  // 2 - panel_w // 2
        panel_y = config.WINDOW_HEIGHT // 2 - panel_h // 2

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((10, 12, 20, 220))
        self._screen.blit(panel_surf, (panel_x, panel_y))

        pygame.draw.rect(self._screen, (220, 170, 30),
                         (panel_x, panel_y, panel_w, panel_h), 3)

        hx = panel_x + panel_w // 2 - headline_surf.get_width() // 2
        hy = panel_y + padding
        self._blit_text_shadowed(self._screen, self._font_large,
                                 headline, (255, 240, 80), (hx, hy))

        sx = panel_x + panel_w // 2 - subline_surf.get_width() // 2
        sy = hy + headline_surf.get_height() + padding // 2
        subline_color = ((0, 229, 255) if reset_mode else (200, 200, 210))
        self._blit_text_shadowed(self._screen, self._font_small,
                                 subline, subline_color, (sx, sy))

    def _draw_start_screen(self):
        """
        Render the welcome / title screen shown before each run.

        Layout (top → bottom):
            Atmospheric maze backdrop (environment drawn at reduced opacity)
            "EVOLUTIONARY SWARM"  large neon-cyan glowing title
            "— Museum Exhibition · JOSEPHS —"  dim subtitle
            "Exhibition Record: X"  gold text (only if a highscore exists)
            "Click or Touch Screen to Begin"  glowing prompt
        """
        self._screen.fill(config.COLOR_BG)

        # Draw the live maze as an atmospheric background
        self._environment.draw(self._screen)

        # Semi-transparent overlay to keep focus on the text
        overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
                                 pygame.SRCALPHA)
        overlay.fill((*config.COLOR_BG, 155))
        self._screen.blit(overlay, (0, 0))

        cx = config.WINDOW_WIDTH  // 2
        cy = config.WINDOW_HEIGHT // 2

        # --- Main title -------------------------------------------------------
        title      = "EVOLUTIONARY SWARM"
        title_surf = self._font_title.render(title, True, config.COLOR_ANT_ALIVE)
        title_x    = cx - title_surf.get_width()  // 2
        title_y    = cy - title_surf.get_height() - 70

        self._blit_text_glowing(self._screen, self._font_title,
                                title, config.COLOR_ANT_ALIVE,
                                (0, 65, 95), (title_x, title_y))

        # --- Venue subtitle ---------------------------------------------------
        venue      = "\u2014  Museum Exhibition  \u00b7  JOSEPHS  \u2014"
        venue_surf = self._font_large.render(venue, True, config.COLOR_TEXT_DIM)
        venue_x    = cx - venue_surf.get_width() // 2
        venue_y    = title_y + title_surf.get_height() + 8
        self._blit_text_shadowed(self._screen, self._font_large,
                                 venue, config.COLOR_TEXT_DIM, (venue_x, venue_y))

        # --- Exhibition highscore (shown only after a winner has been found) -----
        record_height = 0
        if self._exhibition_highscore < 999:
            hs_label      = "Exhibition Record:"
            hs_value      = f"{self._exhibition_highscore} Gens"
            hs_label_surf = self._font_large.render(hs_label, True,
                                                    config.COLOR_TEXT_DIM)
            hs_value_surf = self._font_large.render(hs_value, True,
                                                    config.COLOR_ANT_BEST)
            gap           = 8
            hs_total_w    = (hs_label_surf.get_width() + gap
                             + hs_value_surf.get_width())
            hs_x          = cx - hs_total_w // 2
            hs_y          = venue_y + venue_surf.get_height() + 22
            self._blit_text_shadowed(self._screen, self._font_large,
                                     hs_label, config.COLOR_TEXT_DIM,
                                     (hs_x, hs_y))
            self._blit_text_shadowed(self._screen, self._font_large,
                                     hs_value, config.COLOR_ANT_BEST,
                                     (hs_x + hs_label_surf.get_width() + gap, hs_y))
            record_height = hs_value_surf.get_height() + 22

        # --- "Click to begin" prompt ------------------------------------------
        prompt      = "Click or Touch Screen to Begin"
        prompt_surf = self._font_large.render(prompt, True,
                                              config.COLOR_TEXT_PRIMARY)
        prompt_x    = cx - prompt_surf.get_width() // 2
        prompt_y    = venue_y + venue_surf.get_height() + record_height + 30
        self._blit_text_glowing(self._screen, self._font_large,
                                prompt, config.COLOR_TEXT_PRIMARY,
                                (0, 45, 65), (prompt_x, prompt_y))

        pygame.display.flip()

    # -----------------------------------------------------------------------
    # Main Run Loop  —  State Machine
    # -----------------------------------------------------------------------

    async def run(self):
        """
        Launch the exhibition loop.

        A single async while-loop drives the state machine.  One iteration of
        the loop equals one Pygame frame.  `await asyncio.sleep(0)` at the end
        of every frame yields to the browser event loop (required by Pygbag).

        State transitions are triggered by:
            - pygame.MOUSEBUTTONDOWN / pygame.FINGERDOWN  (visitor input)
            - Elapsed time comparisons  (timed state auto-advances)
            - Simulation conditions  (all ants done, winner detected)
        """
        population     = self._ga.population
        state          = AppState.START_SCREEN
        state_entry_ms = pygame.time.get_ticks()
        eval_best_idx  = -1   # best ant index used during EVALUATING render
        winner_idx     = -1   # winner ant index used during SUCCESS/RESET_LOOP
        winner_gen     = 0    # generation number when a winner was found
        running        = True

        # pygame.FINGERDOWN is available in Pygame 2.x for touch-screen support.
        # getattr with a fallback prevents AttributeError on older builds.
        _finger_down = getattr(pygame, "FINGERDOWN", None)

        while running:
            now = pygame.time.get_ticks()

            # ----------------------------------------------------------------
            # Event handling — processed first, common to all states
            # ----------------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break

                # Unified click / touch detection
                is_input = (
                    event.type == pygame.MOUSEBUTTONDOWN or
                    (_finger_down is not None and event.type == _finger_down)
                )

                if is_input:
                    if state == AppState.START_SCREEN:
                        # Visitor starts a new run
                        self._generation += 1
                        self._spawn_generation(population)
                        state          = AppState.SIMULATING
                        state_entry_ms = now

                    elif state in (AppState.SUCCESS, AppState.RESET_LOOP):
                        # Full reset — re-initialise everything except highscore
                        self._init_run_state()
                        population     = self._ga.population
                        state          = AppState.START_SCREEN
                        state_entry_ms = now

            if not running:
                break

            # ----------------------------------------------------------------
            # State logic + rendering
            # ----------------------------------------------------------------

            if state == AppState.START_SCREEN:
                # Keep the target zone pulsing in the background
                self._environment.update()
                self._draw_start_screen()

            elif state == AppState.SIMULATING:
                # Advance every living ant by one DNA gene this frame
                for ant in self._ants:
                    if ant.is_alive:
                        ant.update(self._environment)

                # Check termination conditions before rendering
                if self._generation_has_winner():
                    self._compute_and_cache_fitness()
                    winner_idx     = self._find_winner_index()
                    winner_gen     = self._generation
                    # Update exhibition highscore: fewest generations to solve.
                    # Only updated here — on SUCCESS entry — never during normal evolution.
                    if self._generation < self._exhibition_highscore:
                        self._exhibition_highscore = self._generation
                    state          = AppState.SUCCESS
                    state_entry_ms = now

                elif self._all_ants_done():
                    self._compute_and_cache_fitness()
                    eval_best_idx  = self._get_best_ant_index()
                    state          = AppState.EVALUATING
                    state_entry_ms = now

                self._environment.update()
                self._render_frame(-1, active_phase=True)

            elif state == AppState.EVALUATING:
                # Show the best ant in gold for 1.5 seconds, then evolve
                self._draw_evaluation_scene(eval_best_idx)
                pygame.display.flip()

                if now - state_entry_ms >= 1500:
                    population     = self._evolve_population()
                    population     = self._check_stagnation_and_inject(population)
                    self._generation += 1
                    self._spawn_generation(population)
                    state          = AppState.SIMULATING
                    state_entry_ms = now

            elif state == AppState.SUCCESS:
                # Victory banner for 5 seconds, then shift to reset prompt
                self._environment.update()
                self._draw_scene(winner_idx)
                self._draw_win_overlay(winner_gen, reset_mode=False)
                pygame.display.flip()

                if now - state_entry_ms >= 5000:
                    state          = AppState.RESET_LOOP
                    state_entry_ms = now

            elif state == AppState.RESET_LOOP:
                # Persistent "touch to reset" prompt until visitor interacts
                self._environment.update()
                self._draw_scene(winner_idx)
                self._draw_win_overlay(winner_gen, reset_mode=True)
                pygame.display.flip()

            self._clock.tick(config.FPS)
            await asyncio.sleep(0)   # yield to browser every frame

        pygame.quit()