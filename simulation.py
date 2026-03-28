"""
simulation.py — Orchestrates the full evolutionary loop.

Architecture overview
---------------------
The SimulationEngine owns:
    - A Pygame window and rendering pipeline.
    - An Environment (maze, zones).
    - A list of Ant agents representing the current generation.
    - A _GeneticAlgorithm instance that manages selection, crossover, mutation.

Generational cycle (outer loop in run()):
    1. Spawn one Ant per genome in the current population.
    2. Step all living ants forward one gene per Pygame frame (inner loop).
    3. After all ants are done, call compute_fitness() on each.
    4. Pass fitness scores directly to _GeneticAlgorithm.evolve().
    5. evolve() returns the offspring population for the next round.
    6. Repeat from step 1.

PyGAD → pure-Python GA
-----------------------
PyGAD is not available in the Pyodide/Pygbag WASM environment, so it has
been replaced by _GeneticAlgorithm — a self-contained class that replicates
the same selection/crossover/mutation behaviour using only the standard
library `random` module.  No third-party packages are required, which makes
the code compatible with both the desktop CPython interpreter and the WASM
Pyodide runtime without any special package loading.

Pygbag / asyncio note
---------------------
run(), _show_evaluation_pause(), and _show_victory_screen() are all declared
async. Each loop body ends with `await asyncio.sleep(0)` to yield control
back to the browser event loop every frame — the required Pygbag pattern.
"""

import asyncio
import math
import random
import pygame

import config
from environment import Environment
from agent import Ant


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

    Private attributes follow the project naming convention
    (single-underscore prefix for internal state):
        _screen             Pygame display surface.
        _clock              Pygame clock for frame-rate control.
        _environment        The maze / zone / wall scene.
        _ants               Active ant population for the current generation.
        _generation         Counter incremented after each generational cycle.
        _best_fitness_ever  Highest fitness score observed across all generations.
        _best_fitness_gen   Highest fitness score in the current generation.
        _max_distance       Straight-line start-to-target distance (normalisation).
        _cached_fitness     Dict[ant_index → fitness] populated after each generation.
        _ga                 _GeneticAlgorithm instance managing the population.
        _lucky_few          Number of elite parents carried over unchanged (elitism).
        _chance_of_mutation Percentage of genes mutated each generation.
        _stagnation_count   Consecutive generations with negligible fitness gain.
        _prev_best_fitness  Best fitness at the last generation where gain was > 5 pts.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self._screen     = pygame.display.set_mode((config.WINDOW_WIDTH,
                                                    config.WINDOW_HEIGHT))
        self._clock      = pygame.time.Clock()
        self._font_large = pygame.font.SysFont("monospace", 26, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 15)

        # Fall back to the built-in default font if monospace is unavailable
        # (SysFont returns None on some WASM builds when the family is missing)
        if self._font_large is None:
            self._font_large = pygame.font.Font(None, 32)
        if self._font_small is None:
            self._font_small = pygame.font.Font(None, 20)

        self._environment = Environment()
        self._ants        = []

        # Evolutionary state counters
        self._generation        = 0
        self._best_fitness_ever = 0.0
        self._best_fitness_gen  = 0.0

        # Stagnation tracking — if the best fitness does not improve by more than
        # 5 points for 8 consecutive generations, _inject_fresh_blood() replaces
        # the weakest 30 % of the population with fresh random genomes.  This
        # prevents the EA from getting permanently stuck in a local optimum, which
        # is critical for a museum showcase that must reliably reach the goal.
        self._stagnation_count  = 0
        self._prev_best_fitness = 0.0

        # EA hyper-parameters
        self._lucky_few          = config.NUM_PARENTS_MATING
        self._chance_of_mutation = config.MUTATION_PERCENT_GENES

        # Pre-compute straight-line start → target distance for fitness normalisation
        sx, sy = self._environment.start_position
        tx, ty = self._environment.target_zone.position
        self._max_distance = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)

        # Fitness cache populated after each visual generation
        self._cached_fitness = {}

        # Initialise the genetic algorithm with a random starting population
        self._ga = _GeneticAlgorithm(
            population_size      = config.POPULATION_SIZE,
            num_genes            = config.DNA_LENGTH,
            gene_low             = -1.0,
            gene_high            =  1.0,
            num_parents_mating   = self._lucky_few,
            mutation_percent_genes = self._chance_of_mutation,
            keep_parents         = 2,
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
        n             = len(self._ga.population)
        fitness_list  = [self._cached_fitness.get(i, 0.0) for i in range(n)]
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
        self._ga.population   = sorted_pop
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
        This is the termination condition for the inner per-generation loop.
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

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _draw_hud(self, alive_count: int):
        """
        Render the heads-up display panel in the top-left corner.

        Shows: generation number, all-time best fitness, current-gen best fitness,
        and number of still-living ants so visitors can track evolutionary progress.
        """
        lines = [
            ("Generation",  str(self._generation)),
            ("Best (ever)", f"{self._best_fitness_ever:,.1f}"),
            ("Best (gen)",  f"{self._best_fitness_gen:,.1f}"),
            ("Alive",       f"{alive_count} / {len(self._ants)}"),
        ]

        padding     = 10
        line_height = 22
        panel_w     = 270
        panel_h     = len(lines) * line_height + 2 * padding

        hud_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        hud_surf.fill((*config.COLOR_HUD_BG, 190))
        self._screen.blit(hud_surf, (8, 8))

        for i, (label, value) in enumerate(lines):
            label_surf = self._font_small.render(f"{label}: ", True,
                                                 config.COLOR_TEXT_DIM)
            value_surf = self._font_small.render(value, True,
                                                 config.COLOR_TEXT_PRIMARY)
            y_pos = 8 + padding + i * line_height
            self._screen.blit(label_surf, (8 + padding, y_pos))
            self._screen.blit(value_surf, (8 + padding + label_surf.get_width(), y_pos))

    def _draw_zone_labels(self):
        """Render 'S' and 'T' text labels centred on the start and target zones."""
        sx, sy = self._environment.start_position
        tx, ty = self._environment.target_zone.position
        for text, x, y in [("S", sx, sy), ("T", tx, ty)]:
            surf = self._font_small.render(text, True, (255, 255, 255))
            self._screen.blit(surf, (x - surf.get_width() // 2,
                                     y - surf.get_height() // 2))

    def _draw_generation_banner(self):
        """
        Render a large centred generation counter at the top of the screen.
        This is the element showcase visitors notice first from a distance.
        """
        text  = f"Generation  {self._generation}"
        surf  = self._font_large.render(text, True, config.COLOR_TEXT_PRIMARY)
        x_pos = config.WINDOW_WIDTH // 2 - surf.get_width() // 2
        self._screen.blit(surf, (x_pos, 6))

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

        if active_phase:
            for ant in self._ants:
                ant.draw(self._screen, active_uniform=True)
        else:
            has_known_best = (best_ant_index >= 0)

            # Dead non-best ants first (rendered behind living ants)
            for ant in self._ants:
                is_this_the_best = has_known_best and ant.index == best_ant_index
                if not ant.is_alive and not is_this_the_best:
                    ant.draw(self._screen, is_best=False)

            # Living ants
            for ant in self._ants:
                if ant.is_alive:
                    ant.draw(self._screen,
                             is_best=(has_known_best and ant.index == best_ant_index))

            # Confirmed best / winner drawn last so it sits on top of everything
            if has_known_best and self._ants:
                best_ant = self._ants[best_ant_index]
                if not best_ant.is_alive:
                    best_ant.draw(self._screen, is_best=True)

        self._draw_zone_labels()
        alive_count = sum(1 for a in self._ants if a.is_alive)
        self._draw_hud(alive_count)
        self._draw_generation_banner()

    def _render_frame(self, best_ant_index: int, active_phase: bool = False):
        """Draw the scene and flip the display — the normal per-frame call."""
        self._draw_scene(best_ant_index, active_phase=active_phase)
        pygame.display.flip()

    def _draw_win_overlay(self, winning_generation: int):
        """
        Composite a centred victory banner on top of whatever is already drawn
        on self._screen. Does NOT call pygame.display.flip() — the caller flips
        after this so the banner and scene appear in a single atomic update.
        """
        headline = f"Target Reached in Generation {winning_generation}!"
        subline  = "Press ESC or close the window to exit."

        headline_surf = self._font_large.render(headline, True, (255, 240, 80))
        subline_surf  = self._font_small.render(subline,  True, (200, 200, 210))

        padding  = 28
        panel_w  = max(headline_surf.get_width(),
                       subline_surf.get_width()) + padding * 2
        panel_h  = (headline_surf.get_height()
                    + subline_surf.get_height()
                    + padding * 3)
        panel_x  = config.WINDOW_WIDTH  // 2 - panel_w // 2
        panel_y  = config.WINDOW_HEIGHT // 2 - panel_h // 2

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((10, 12, 20, 220))
        self._screen.blit(panel_surf, (panel_x, panel_y))

        pygame.draw.rect(self._screen, (220, 170, 30),
                         (panel_x, panel_y, panel_w, panel_h), 3)

        hx = panel_x + panel_w // 2 - headline_surf.get_width() // 2
        hy = panel_y + padding
        self._screen.blit(headline_surf, (hx, hy))

        sx = panel_x + panel_w // 2 - subline_surf.get_width() // 2
        sy = hy + headline_surf.get_height() + padding // 2
        self._screen.blit(subline_surf, (sx, sy))

    def _generation_has_winner(self) -> bool:
        """
        Returns True if any ant in the current generation has set _has_won.
        Called once per frame inside the inner simulation loop so the engine
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

    def _draw_evaluation_scene(self, best_ant_index: int):
        """
        Render the post-generation evaluation frame.

        All ants except the confirmed best are drawn as dim dead dots (no trail).
        The best ant is drawn in gold with a thick trail so visitors can trace
        the exact path the EA judged to be strongest this generation.
        """
        self._screen.fill(config.COLOR_BG)
        self._environment.draw(self._screen)

        has_known_best = (best_ant_index >= 0)

        for ant in self._ants:
            is_this_the_best = has_known_best and ant.index == best_ant_index
            if not is_this_the_best:
                ant.draw(self._screen, is_best=False)

        if has_known_best and self._ants:
            best_ant = self._ants[best_ant_index]
            best_ant.draw(self._screen, is_best=True, trail_width=3)

        self._draw_zone_labels()
        alive_count = sum(1 for a in self._ants if a.is_alive)
        self._draw_hud(alive_count)
        self._draw_generation_banner()

    async def _show_evaluation_pause(self, best_ant_index: int) -> bool:
        """
        Display the evaluation frame and hold for 1.5 seconds.

        Processes events during the wait so the window stays responsive.
        Returns False if the user quit, True otherwise.
        Uses await asyncio.sleep(0.05) to yield to the browser each slice.
        """
        self._draw_evaluation_scene(best_ant_index)
        pygame.display.flip()

        elapsed = 0
        while elapsed < 1500:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
            await asyncio.sleep(0.05)
            elapsed += 50

        return True

    async def _show_victory_screen(self, winner_idx: int):
        """
        Freeze the simulation in a victory display until the user exits.
        The winning ant's full trail stays visible; the target zone keeps
        pulsing. Evolution does not continue after this method returns.
        """
        winning_generation = self._generation
        display_running    = True

        while display_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    display_running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    display_running = False

            self._environment.update()
            self._draw_scene(winner_idx)
            self._draw_win_overlay(winning_generation)
            pygame.display.flip()
            self._clock.tick(config.FPS)
            await asyncio.sleep(0)

    # -----------------------------------------------------------------------
    # Main Run Loop
    # -----------------------------------------------------------------------

    async def run(self):
        """
        Launch the simulation.

        Outer loop  : one iteration = one evolutionary generation.
        Inner loop  : one iteration = one Pygame frame / one DNA step per ant.

        Declared async for Pygbag. Each frame ends with await asyncio.sleep(0)
        to yield control to the browser — without this the tab freezes for the
        entire duration of a generation.
        """
        population = self._ga.population
        running    = True

        while running:
            self._generation += 1
            best_idx = -1
            self._spawn_generation(population)

            # --- Inner loop: animate the current generation ------------------
            while running and not self._all_ants_done():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                for ant in self._ants:
                    if ant.is_alive:
                        ant.update(self._environment)

                if self._generation_has_winner():
                    break

                self._environment.update()
                self._render_frame(-1, active_phase=True)
                self._clock.tick(config.FPS)
                await asyncio.sleep(0)   # yield to browser every frame

            if not running:
                break

            # --- Post-generation: winner check before evolving ---------------
            if self._generation_has_winner():
                self._compute_and_cache_fitness()
                winner_idx = self._find_winner_index()
                await self._show_victory_screen(winner_idx)
                break

            # --- Normal post-generation: evaluate fitness, evolve ------------
            self._compute_and_cache_fitness()
            best_idx   = self._get_best_ant_index()
            running    = await self._show_evaluation_pause(best_idx)

            if not running:
                break

            population = self._evolve_population()

            # Detect stagnation and inject fresh genomes if the EA is stuck.
            # Must run AFTER _evolve_population so the fresh individuals enter
            # the next generation rather than being immediately overwritten.
            population = self._check_stagnation_and_inject(population)

        pygame.quit()
