from pypokerengine.api.game import setup_config, start_poker
from random_player import FishPlayer
from learned_player import BotPlayer


def main():
    scores = [0, 0]
    for _ in range(10):
        config = setup_config(max_round=1000, initial_stack=100, small_blind_amount=5)
        config.register_player(name="p1", algorithm=BotPlayer())
        config.register_player(name="p2", algorithm=BotPlayer())
        config.register_player(name="p3", algorithm=FishPlayer())
        config.register_player(name="p4", algorithm=FishPlayer())
        game_result = start_poker(config, verbose=0)
        winner = [x for x in game_result['players'] if x['state'] == 'participating'][0]
        if winner['name'] in ('p1', 'p2'):
            scores[0] += 1
        else:
            scores[1] += 1
        print(f"{winner['name']} won! {scores}")


if __name__ == '__main__':
    main()
