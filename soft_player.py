from learned_player import BotPlayer


def setup_ai():
    return BotPlayer({
        'call': 0.0,  # 0 - call if win is predicted , 1 - always calls
        'raise': 0.0,  # 0 - never, 1 - always
        'allin': 0.01  # 0 - never 1 - always
    })
