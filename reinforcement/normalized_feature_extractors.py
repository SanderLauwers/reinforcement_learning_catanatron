from catanatron_gym.features import *
from catanatron.models.map import *

def nzd_player_features(game: Game, p0_color: Color):
	# P0_ACTUAL_VPS
	# P{i}_PUBLIC_VPS, P1_PUBLIC_VPS, ...
	# P{i}_HAS_ARMY, P{i}_HAS_ROAD, P1_HAS_ARMY, ...
	# P{i}_ROADS_LEFT, P{i}_SETTLEMENTS_LEFT, P{i}_CITIES_LEFT, P1_...
	# P{i}_HAS_ROLLED, P{i}_LONGEST_ROAD_LENGTH
	features = dict()
	vptw = game.vps_to_win

	for i, color in iter_players(game.state.colors, p0_color):
		key = player_key(game.state, color)
		if color == p0_color:
			features["P0_ACTUAL_VPS"] = game.state.player_state[
				key + "_ACTUAL_VICTORY_POINTS"
			] * 2 / vptw - 1

		features[f"P{i}_PUBLIC_VPS"] = game.state.player_state[key + "_VICTORY_POINTS"] * 2 / vptw - 1
		features[f"P{i}_HAS_ARMY"] = game.state.player_state[key + "_HAS_ARMY"]
		features[f"P{i}_HAS_ROAD"] = game.state.player_state[key + "_HAS_ROAD"]
		features[f"P{i}_ROADS_LEFT"] = game.state.player_state[key + "_ROADS_AVAILABLE"] * 2 / 15 - 1
		features[f"P{i}_SETTLEMENTS_LEFT"] = game.state.player_state[key + "_SETTLEMENTS_AVAILABLE"] * 2 / 5 - 1
		features[f"P{i}_CITIES_LEFT"] = game.state.player_state[key + "_CITIES_AVAILABLE"] * 2 / 4 - 1
		features[f"P{i}_HAS_ROLLED"] = game.state.player_state[key + "_HAS_ROLLED"]
		features[f"P{i}_LONGEST_ROAD_LENGTH"] = (game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]) * 2 / 5 - 1

	return features

def nzd_resource_hand_features(game: Game, p0_color: Color):
	# P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
	# P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
	# P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
	# P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

	# P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...
	# TODO: P1_WHEATS_INFERENCE, P1_WOODS_INFERENCE, ...
	# TODO: P1_ROAD_BUILDINGS_INFERENCE, P1_KNIGHT_INFERENCE, ...

	state = game.state
	player_state = state.player_state

	features = {}
	for i, color in iter_players(game.state.colors, p0_color):
		key = player_key(game.state, color)

		if color == p0_color:
			for resource in RESOURCES:
				features[f"P0_{resource}_IN_HAND"] = player_state[key + f"_{resource}_IN_HAND"] / 5
			for card in DEVELOPMENT_CARDS:
				features[f"P0_{card}_IN_HAND"] = player_state[key + f"_{card}_IN_HAND"] / 3
			features[f"P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = player_state[
				key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
			]

		for card in DEVELOPMENT_CARDS:
			if card == VICTORY_POINT:
				continue  # cant play VPs
			features[f"P{i}_{card}_PLAYED"] = player_state[key + f"_PLAYED_{card}"] / 6

		features[f"P{i}_NUM_RESOURCES_IN_HAND"] = player_num_resource_cards(
			state, color
		) / 7
		features[f"P{i}_NUM_DEVS_IN_HAND"] = player_num_dev_cards(state, color) / 4

	return features


def number_probability(number):
    return DICE_PROBAS[number]

@functools.lru_cache(NUM_TILES * 2)  # one for each robber, and acount for Minimap
def nzd_map_tile_features(catan_map: CatanMap, robber_coordinate):
	# Returns list of functions that take a game and output a feature.
	# build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
	features = {}

	for tile_id, tile in catan_map.tiles_by_id.items():
		for resource in RESOURCES:
			features[f"TILE{tile_id}_IS_{resource}"] = tile.resource == resource
		features[f"TILE{tile_id}_IS_DESERT"] = tile.resource == None
		features[f"TILE{tile_id}_PROBA"] = (
			0 if tile.resource is None else number_probability(tile.number)
		)
		features[f"TILE{tile_id}_HAS_ROBBER"] = (
			catan_map.tiles[robber_coordinate] == tile
		)
	return features

def nzd_tile_features(game: Game, p0_color: Color):
	# Returns list of functions that take a game and output a feature.
	# build features like tile0_is_wood, tile0_is_wheat, ..., tile0_proba, tile0_hasrobber
	return map_tile_features(game.state.board.map, game.state.board.robber_coordinate)


@functools.lru_cache(1)
def nzd_map_port_features(catan_map):
	features = {}
	for port_id, port in catan_map.ports_by_id.items():
		for resource in RESOURCES:
			features[f"PORT{port_id}_IS_{resource}"] = port.resource == resource
		features[f"PORT{port_id}_IS_THREE_TO_ONE"] = port.resource is None
	return features

def nzd_port_features(game, p0_color):
	# PORT0_WOOD, PORT0_THREE_TO_ONE, ...
	return map_port_features(game.state.board.map)


@functools.lru_cache(4)
def nzd_initialize_graph_features_template(num_players, catan_map: CatanMap):
	features = {}
	for i in range(num_players):
		for node_id in range(len(catan_map.land_nodes)):
			for building in [SETTLEMENT, CITY]:
				features[f"NODE{node_id}_P{i}_{building}"] = False
		for edge in get_edges(catan_map.land_nodes):
			features[f"EDGE{edge}_P{i}_ROAD"] = False
	return features

@functools.lru_cache(1024 * 2 * 2 * 2)
def nzd_get_node_hot_encoded(player_index, colors, settlements, cities, roads):
	features = {}

	for node_id in settlements:
		features[f"NODE{node_id}_P{player_index}_SETTLEMENT"] = True
	for node_id in cities:
		features[f"NODE{node_id}_P{player_index}_CITY"] = True
	for edge in roads:
		features[f"EDGE{tuple(sorted(edge))}_P{player_index}_ROAD"] = True

	return features

def nzd_graph_features(game: Game, p0_color: Color):
	features = initialize_graph_features_template(
		len(game.state.colors), game.state.board.map
	).copy()

	for i, color in iter_players(game.state.colors, p0_color):
		settlements = tuple(game.state.buildings_by_color[color][SETTLEMENT])
		cities = tuple(game.state.buildings_by_color[color][CITY])
		roads = tuple(game.state.buildings_by_color[color][ROAD])
		to_update = get_node_hot_encoded(
			i, game.state.colors, settlements, cities, roads
		)
		features.update(to_update)

	return features


def nzd_game_features(game: Game, p0_color: Color):
	# BANK_WOODS, BANK_WHEATS, ..., BANK_DEV_CARDS
	possibilities = set([a.action_type for a in game.state.playable_actions])
	features = {
		"BANK_DEV_CARDS": len(game.state.development_listdeck) * 2 / 25 - 1,
		"IS_MOVING_ROBBER": ActionType.MOVE_ROBBER in possibilities,
		"IS_DISCARDING": ActionType.DISCARD in possibilities,
	}
	for resource in RESOURCES:
		features[f"BANK_{resource}"] = freqdeck_count(
			game.state.resource_freqdeck, resource
		) * 2 / 19 - 1
	return features


nzd_feature_extractors = [
	# PLAYER FEATURES =====
	nzd_player_features,
	nzd_resource_hand_features,
	# RAW BASE-MAP FEATURES =====
	nzd_tile_features,
	nzd_port_features,
	nzd_graph_features,
	# GAME FEATURES =====
	nzd_game_features,
]