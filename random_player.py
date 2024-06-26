from pypokerengine.players import BasePokerPlayer
from random import choice, randint


class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        if valid_actions[2]['amount']['max'] == -1:
            return 'fold', 0
        action = choice(valid_actions)
        if action['action'] == 'raise':
            amount = randint(action['amount']['min'], action['amount']['max'])
        else:
            amount = action["amount"]
        action, amount = action["action"], amount
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return FishPlayer()
