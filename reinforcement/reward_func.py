from catanatron import Color
from catanatron.models.enums import (
    VICTORY_POINT,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    SETTLEMENT,
    CITY,
    ROAD
)
from catanatron.models.map import get_nodes_and_edges
from catanatron.models.coordinate_system import add


# https://catanatron.readthedocs.io/en/latest/catanatron_gym.envs.html
# game.state https://catanatron.readthedocs.io/en/latest/catanatron.html#module-catanatron.state
def reward_function(game, p0_color):
	index_self = str(game.state.current_player_index)
	index_enemy =  "1" if index_self == "0" else "0"
	color_self =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_self))]
	color_enemy =  list(game.state.color_to_index.keys())[list(game.state.color_to_index.values()).index(int(index_enemy))]
	
	p_state = game.state.player_state
	board = game.state.board

	# is winner
	winning_color = game.winning_color()
	if p0_color == winning_color:
		return 1
	elif winning_color is not None:
		return -1

	reward = 0

	# + for each possessed resource, but - if too many; - for enemy cards
	resource_amount = p_state["P"+index_self+"_WOOD_IN_HAND"] + p_state["P"+index_self+"_BRICK_IN_HAND"] + p_state["P"+index_self+"_SHEEP_IN_HAND"] + p_state["P"+index_self+"_WHEAT_IN_HAND"] + p_state["P"+index_self+"_ORE_IN_HAND"]
	reward += resource_amount
	if (resource_amount > 7):
		reward -= 2*(resource_amount - 7)

	enemy_resource_amount = p_state["P"+index_enemy+"_WOOD_IN_HAND"] + p_state["P"+index_enemy+"_BRICK_IN_HAND"] + p_state["P"+index_enemy+"_SHEEP_IN_HAND"] + p_state["P"+index_enemy+"_WHEAT_IN_HAND"] + p_state["P"+index_enemy+"_ORE_IN_HAND"]
	reward -= enemy_resource_amount

	# + for dev cards
	reward += 2*(p_state["P"+index_self+"_KNIGHT_IN_HAND"] + p_state["P"+index_self+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_self+"_MONOPOLY_IN_HAND"] + p_state["P"+index_self+"_ROAD_BUILDING_IN_HAND"])
	reward -= 2*(p_state["P"+index_enemy+"_KNIGHT_IN_HAND"] + p_state["P"+index_enemy+"_YEAR_OF_PLENTY_IN_HAND"] + p_state["P"+index_enemy+"_MONOPOLY_IN_HAND"] + p_state["P"+index_enemy+"_ROAD_BUILDING_IN_HAND"])
	reward += 3*p_state["P"+index_self+"_PLAYED_KNIGHT"]
	reward -= 3*p_state["P"+index_enemy+"_PLAYED_KNIGHT"]

	# + for robber on enemy tile, - for robber on owned tile
	robbed_nodes = board.map.tiles[board.robber_coordinate].nodes
	for node_id in robbed_nodes.values():
		if not node_id in list(board.buildings.keys()):
			continue
		robbed_building = board.buildings[node_id]
		reward += -4 if robbed_building[0] == color_self else 4
		
	# + per vp difference
	reward += 10 * (p_state["P"+index_self+"_ACTUAL_VICTORY_POINTS"] - p_state["P"+index_enemy+"_VICTORY_POINTS"])

	# +++ per city & ++ per village & + per road; city and village already in vp?
	reward += len(game.state.buildings_by_color[color_self][ROAD])
	reward += 2*len(game.state.buildings_by_color[color_self][SETTLEMENT])
	reward += 4*len(game.state.buildings_by_color[color_self][CITY])

	reward -= len(game.state.buildings_by_color[color_enemy][ROAD])
	reward -= 2*len(game.state.buildings_by_color[color_enemy][SETTLEMENT])
	reward -= 4*len(game.state.buildings_by_color[color_enemy][CITY])

	return reward / 100