"""
Cogitive Player
==========================

Cognitive Science Grounding (Thagard, MIND ch. 3)
--------------------------------------------------
Expert players shift goals based on game phase — a direct application of
SOAR's goal-directed heuristic search. In Catan priorities could look like:
    Early game: production diversity matters most. 
    Mid game: city-building becomes the priority. 
    Late game: every decision is evaluated purely against VP acquisition.

This is implemented as a DYNAMIC VALUE FUNCTION that re-weights the
existing catanatron heuristic features based on the current game phase,
rather than the static DEFAULT_WEIGHTS throughout the entire game.

The implementation is intentionally minimal:
  - Inherits AlphaBetaPlayer directly (utilize the same search algorithm, same depth=2, but override the value funtion)
  - Only overrides value_function()
  - Cleanly comparable to baseline (same everything except weights)
"""

import math
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn
from catanatron.state_functions import (
    player_key,
    player_num_resource_cards,
    get_longest_road_length,
    get_played_dev_cards,
    player_num_dev_cards,
)
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
from catanatron.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)
from catanatron.players.value import value_production

# ____ game phase detection _______________________________________________
def _vp(state, color) -> int:
    key = player_key(state, color)
    return int(state.player_state.get(f"{key}_VICTORY_POINTS", 0))

# def _num_cities(state, color) -> int:
#     return len(state.buildings_by_color[color].get(CITY, []))

# def _num_settlements(state, color) -> int:
#     return len(state.buildings_by_color[color].get(SETTLEMENT, []))
    
def get_phase(game: Game, color) -> str:
    """
    Classify current game phase, incorporating Theory of Mind (opponent modeling).
    """
    my_vp = _vp(game.state, color)
    
    # 1. Theory of Mind check: Is anyone else about to win?
    for c in game.state.colors:
        if c != color:
            enemy_vp = _vp(game.state, c)
            # If an opponent is 2 points away from winning, drop everything and block.
            if enemy_vp >= 8:
                return "THREAT"

    # 2. Otherwise, rely on our own goal stack
    if my_vp < 5:
        return "EARLY"
    elif my_vp < 8:
        return "MID"
    else:
        return "LATE"

# ____ phase-specific weight sets ________________________________________________
# Weights are expressed as simple multiples of P = production = 1e8.
# Every weight has a single justifying IF-THEN rule.
# This makes the cognitive science claim explicit and auditable:
# the SAME features are weighted differently because different GOALS are active.
#
# Notation: P = 1e8 (production anchor, same as DEFAULT_WEIGHTS)
# Ratio     Meaning
# 0.0x P    irrelevant to this phase's goal
# 0.1x P    minor consideration
# 0.5x P    secondary goal
# 1.0x P    co-equal with production
# 5.0x P    dominates production

P = 1e8  # production anchor

EARLY_WEIGHTS = {
    # Active goal: ESTABLISH PRODUCTION ENGINE + EXPAND NETWORK
    # IF early game THEN production is the primary objective          -> 1.0x P
    # IF early game THEN expansion enables future production          -> 0.5x P
    # IF early game THEN enemy is not yet a threat                    -> 0.1x P
    # IF early game THEN hand composition doesn't matter yet          -> 0.1x P
    "public_vps":              3e14,        # winning always dominates
    "production":              P,           # 1.0x — primary goal
    "enemy_production":        -P * 0.1,    # 0.1x — ignore enemy early
    "num_tiles":               P * 0.0001,
    "reachable_production_0":  P * 0.01,
    "reachable_production_1":  P * 0.001,
    "buildable_nodes":         P * 0.5,     # 0.5x — expansion coequal to production
    "longest_road":            P * 0.0001,
    "hand_synergy":            P * 0.001,
    "hand_resources":          P * 0.00001,
    "discard_penalty":         -P * 0.00005,
    "hand_devs":               P * 0.00001,
    "army_size":               P * 0.00001,
}

MID_WEIGHTS = {
    # Active goal: CONVERT RESOURCES TO CITIES + BLOCK OPPONENT
    # IF mid game THEN hand synergy (ore+wheat) ~ half as important as production -> 0.5x P
    # IF mid game THEN blocking opponent more important than own expansion         -> 1.5x P
    # IF mid game THEN expansion is largely complete                               -> 0.1x P
    # IF mid game THEN dev cards open VP paths                                     -> 0.1x P
    "public_vps":              3e14,
    "production":              P,           # 1.0x — anchor unchanged
    "enemy_production":        -P * 1.5,    # 1.5x — block opponent mid-game
    "num_tiles":               P * 0.00001,
    "reachable_production_0":  0,
    "reachable_production_1":  P * 0.00001,
    "buildable_nodes":         P * 0.005,   # 0.005x — expansion mostly done
    "longest_road":            P * 0.0005,
    "hand_synergy":            P * 0.5,     # 0.5x — hand synergy ~ production
    "hand_resources":          P * 0.0001,
    "discard_penalty":         -P * 0.0002,
    "hand_devs":               P * 0.1,     # 0.1x — dev cards open VP paths
    "army_size":               P * 0.1,
}

LATE_WEIGHTS = {
    # Active goal: WIN THE VP RACE
    # IF late game THEN stopping opponent is top priority             -> 5.0x P
    # IF late game THEN dev cards are direct VP paths                 -> 1.0x P
    # IF late game THEN army/road bonus can be game-winning VP        -> 1.0x P
    # IF late game THEN expansion is completely irrelevant            -> 0.0x P
    "public_vps":              3e14,
    "production":              P,           # 1.0x — still matters
    "enemy_production":        -P * 5,      # 5.0x — stopping opponent = top priority
    "num_tiles":               0,
    "reachable_production_0":  0,
    "reachable_production_1":  0,           # 0.0x — no more expansion
    "buildable_nodes":         0,           # 0.0x — expansion irrelevant
    "longest_road":            P * 1.0,     # 1.0x — road VP = game-winning
    "hand_synergy":            P * 0.01,
    "hand_resources":          P * 0.0002,
    "discard_penalty":         -P * 0.0005,
    "hand_devs":               P * 1.0,     # 1.0x — dev cards = direct VP paths
    "army_size":               P * 1.0,     # 1.0x — army VP = game-winning
}

THREAT_WEIGHTS = {
    # Active goal: SURVIVAL / BLOCK OPPONENT
    # IF opponent is near win THEN own expansion is irrelevant   -> 0.0x P
    # IF opponent is near win THEN reducing their production is  -> 10.0x P
    # IF opponent is near win THEN playing knights/robber is key -> 2.0x P
    "public_vps":              3e14,
    "production":              P * 0.5,     # Still need some production to buy knights
    "enemy_production":        -P * 10.0,   # 10.0x — Maximum priority to rob/block them
    "num_tiles":               0,
    "reachable_production_0":  0,
    "reachable_production_1":  0,
    "buildable_nodes":         0,
    "longest_road":            P * 1.5,     # Only care if we are stealing it from them
    "hand_synergy":            0,
    "hand_resources":          P * 0.0001,
    "discard_penalty":         -P * 0.0005,
    "hand_devs":               P * 0.5,
    "army_size":               P * 2.0,     # High priority to play knights to move robber
}

PHASE_WEIGHTS = {
    "EARLY": EARLY_WEIGHTS,
    "MID":   MID_WEIGHTS,
    "LATE":  LATE_WEIGHTS,
    "THREAT": THREAT_WEIGHTS,
}

# ____ value function __________________________________________________________

def phase_aware_value(game: Game, color) -> float:
    """
    Evaluate the game state using weights appropriate to the current phase.
    Uses the exact same computation as base_fn but with dynamic weights.
    """
    phase = get_phase(game, color)
    params = PHASE_WEIGHTS[phase]

    production_features = build_production_features(True)
    our_sample = production_features(game, color)
    enemy_sample = production_features(game, color)

    production = value_production(our_sample,   "P0")
    enemy_production = value_production(enemy_sample, "P1", False)

    key = player_key(game.state, color)
    longest_road = get_longest_road_length(game.state, color)

    reach = reachability_features(game, color, 2)
    reachable_0 = sum(reach.get(f"P0_0_ROAD_REACHABLE_{r}", 0) for r in RESOURCES)
    reachable_1 = sum(reach.get(f"P0_1_ROAD_REACHABLE_{r}", 0) for r in RESOURCES)

    hand_sample = resource_hand_features(game, color)
    dist_city = (
        max(2 - hand_sample.get("P0_WHEAT_IN_HAND", 0), 0) +
        max(3 - hand_sample.get("P0_ORE_IN_HAND",   0), 0)
    ) / 5.0
    dist_settle = (
        max(1 - hand_sample.get("P0_WHEAT_IN_HAND", 0), 0) +
        max(1 - hand_sample.get("P0_SHEEP_IN_HAND", 0), 0) +
        max(1 - hand_sample.get("P0_BRICK_IN_HAND", 0), 0) +
        max(1 - hand_sample.get("P0_WOOD_IN_HAND",  0), 0)
    ) / 4.0

    hand_synergy = (2 - dist_city - dist_settle) / 2

    num_in_hand = player_num_resource_cards(game.state, color)
    discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

    buildings = game.state.buildings_by_color[color]
    owned_nodes = buildings.get(SETTLEMENT, []) + buildings.get(CITY, [])
    owned_tiles = set()
    for n in owned_nodes:
        owned_tiles.update(game.state.board.map.adjacent_tiles[n])
    num_tiles = len(owned_tiles)

    num_buildable = len(game.state.board.buildable_node_ids(color))
    longest_road_factor = params["longest_road"] if num_buildable == 0 else 0.1

    return float(
        game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
        + production                        * params["production"]
        + enemy_production                  * params["enemy_production"]
        + reachable_0                       * params["reachable_production_0"]
        + reachable_1                       * params["reachable_production_1"]
        + hand_synergy                      * params["hand_synergy"]
        + num_buildable                     * params["buildable_nodes"]
        + num_tiles                         * params["num_tiles"]
        + num_in_hand                       * params["hand_resources"]
        + discard_penalty
        + longest_road                      * longest_road_factor
        + player_num_dev_cards(game.state, color)              * params["hand_devs"]
        + get_played_dev_cards(game.state, color, "KNIGHT")    * params["army_size"]
    )

# ____ player _____________________________________________________________________

class CognitivePlayer(AlphaBetaPlayer):
    """
    AlphaBeta player with a phase-aware, goal-directed value function.

    Cognitive science basis (Thagard, MIND ch. 3):
    Expert problem-solvers shift goals as the problem state changes.
    SOAR models this as a goal stack where active goals determine which
    operators (actions) and evaluations are applied. Here, phase detection
    activates different weight profiles, implementing goal-directed
    evaluation without modifying the search architecture.

    Identical to AlphaBetaPlayer(depth=2, prunning=True) in every way
    except the leaf evaluation function — isolating the hypothesis.
    """

    def __init__(self, color, depth=2, prunning=False, **kwargs):
        super().__init__(color, depth=depth, prunning=prunning, **kwargs)
        self.use_value_function = True   # tells parent to use our override

    def value_function(self, game: Game, color) -> float:
        # override with my phase based value funtion
        return phase_aware_value(game, color)
    
        # TODO move ordering

    def __repr__(self) -> str:
        return (
            super(AlphaBetaPlayer, self).__repr__()
            + f"(depth={self.depth},phase-aware-value,prunning={self.prunning})"
        )
    
    # TODO Put close eval states head to head in a rules based tie breaker