from learned_player import BotPlayer


def setup_ai():
    return BotPlayer(strategy={
        'call': 0.5,  # 0 - call if win is predicted , 1 - always calls
        'raise': 0.5,  # 0 - never, 1 - always
        'allin': 0.02  # 0 - never 1 - always
    })
