"""
SBS — Smart Swarm simulation package.

Modules
-------
config      Central constants: window size, EA parameters, maze layout, colours.
agent       BaseAgent abstract class and the concrete Ant simulation agent.
environment Maze walls, start/target zones, and the Environment scene manager.
simulation  SimulationEngine, _GeneticAlgorithm, and the AppState machine.
run_web     Browser launcher: pre-caches BrowserFS then hands off to Pygbag.
"""