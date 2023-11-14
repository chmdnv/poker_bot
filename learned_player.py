from pypokerengine.players import BasePokerPlayer
from random import choice, randint
from processing import parse_from_data, predict


class BotPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __new__(cls, strategy=None):
        if not strategy:
            strategy = {
                'call': 0.8,   # 0 - call if win is predicted , 1 - always calls
                'raise': 0.5,  # 0 - never, 1 - always
                'allin': 0.01  # 0 - never 1 - always
            }
        self = super().__new__(cls)
        self.strategy = strategy
        return self

    def get_attrs(self, round_state):
        for i, d in enumerate(round_state['seats']):
            if d['uuid'] == self.uuid:
                self.seat = i

        if round_state['small_blind_pos'] == self.seat:
            self.is_smallbl = True
        else:
            self.is_smallbl = False

        if round_state['big_blind_pos'] == self.seat:
            self.is_bigbl = True
        else:
            self.is_bigbl = False

        self.stack = round_state['seats'][self.seat]['stack']

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        self.get_attrs(round_state)

        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        action, amount = predict(
            parse_from_data(round_state, self.uuid, hole_card, valid_actions),
            self.strategy
        )
        amount *= self.stack

        if (
                (self.is_smallbl or self.is_bigbl) and
                round_state['street'] == 'preflop' and
                valid_actions[1]['amount'] >= 0
        ):
            action = 'call'
            amount = valid_actions[1]['amount']

        return action, int(amount)  # action returned here is sent to the poker engine

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
    return BotPlayer()