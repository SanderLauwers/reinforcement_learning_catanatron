from catanatron import Player
from catanatron_experimental.cli.cli_players import register_player
import random

@register_player("OWNRANDOM")
class OwnRandomPlayer(Player):
  def decide(self, game, playable_actions):
    """Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """
    # ===== YOUR CODE HERE =====

    return playable_actions[random.randint(0, len(playable_actions) - 1)]
  
    # ===== END YOUR CODE =====