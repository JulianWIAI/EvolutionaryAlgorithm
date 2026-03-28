"""
main.py — Application entry point for the Smart Swarm simulation.

Usage
-----
    Desktop (standard Python):
        python main.py

    Browser (Pygbag / WebAssembly):
        python -m pygbag .
        then open http://localhost:8000

Dependencies (install via pip):
    pip install pygame numpy pygbag

Controls
--------
    Escape / window close  →  Quit the simulation at any time.

asyncio note
------------
main() is declared async and the entry point uses asyncio.run(main()).
This satisfies Pygbag's requirement that the top-level coroutine be
scheduled through the asyncio event loop, keeping the browser responsive
between frames.
"""

import asyncio
import traceback
import pygame

# Scientific Showcase Project: EvolutionaryAlgorithm
# Developed for the JOSEPHS Exhibition.
# Disclaimer: The architecture, simulation logic, and visual rendering of this
# project were co-developed with Artificial Intelligence (Claude 3.5 Sonnet /
# Gemini) to demonstrate the capabilities of Evolutionary Algorithms.


async def main():
    """
    Instantiate and launch the simulation engine.

    The import of SimulationEngine is placed INSIDE this function so that
    any failure in the import chain (e.g. a missing module in the WASM
    virtual filesystem) is caught by the try/except below.  If it were at
    module level, a crash there would prevent asyncio.run(main()) from ever
    executing and the browser would show a silent grey page with no error.
    """
    try:
        from SBS.simulation import SimulationEngine   # deferred import — errors are catchable here
        engine = SimulationEngine()
        await engine.run()

    except Exception:
        # Print the full traceback to the browser console (F12 → Console)
        traceback.print_exc()

        # Also render the error on screen so it is visible without opening
        # devtools — important at an exhibition where devtools are closed.
        try:
            if not pygame.get_init():
                pygame.init()
            screen = pygame.display.set_mode((900, 600))
            font   = pygame.font.Font(None, 22)
            screen.fill((15, 0, 0))

            lines = traceback.format_exc().splitlines()
            y = 20
            for line in lines:
                if len(line) > 110:
                    line = line[:107] + "..."
                surf = font.render(line, True, (255, 100, 100))
                screen.blit(surf, (20, y))
                y += 22
                if y > 570:
                    break

            pygame.display.flip()
        except Exception:
            pass   # If even the error screen fails, the console has it

        # Keep the window open so the traceback stays readable
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            await asyncio.sleep(0)


asyncio.run(main())
