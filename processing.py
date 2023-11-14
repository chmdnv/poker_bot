import random
import os
import pandas as pd
import numpy as np
import joblib

pd.options.mode.chained_assignment = None  # default='warn'

max_players = 9
# PATH = 'C:\\Users\\Arseniy\\Documents\\Poker\\'

file_name = max([x for x in os.listdir('model/') if x.startswith('win_model') and x.endswith('.pkl')])
meta = joblib.load(f"model/{file_name}")


def parse_from_data(data: dict, bot_uuid: str, bot_hand: list, valid_actions: list[dict]):
    columns = joblib.load('data/parsed5_columns.pkl')
    df = pd.DataFrame(columns=columns)

    df.loc[len(df.index)] = [np.nan] * len(df.columns)
    df.Game_ID.iloc[-1] = int(id(data) // 1e6)
    df.num_active_pl.iloc[-1] = len(data['seats'])
    df.game_stage.iloc[-1] = data['street']
    df.bank.iloc[-1] = data['pot']['main']['amount']
    df.bet.iloc[-1] = valid_actions[1]['amount']

    for i, seat in enumerate(data['seats'], 1):
        df[f"s{i}_money"].iloc[-1] = seat['stack'] if seat['state'] == 'participating' else 0
        if seat['uuid'] == bot_uuid:
            df.bot_seat.iloc[-1] = i
            df.bot_money.iloc[-1] = df[f"s{i}_money"].iloc[-1]

    df.bot_hand.iloc[-1] = ' '.join(bot_hand)[::-1].replace('T', '10') or 'hidden'
    df.board.iloc[-1] = ' '.join(data['community_card'])[::-1].replace('T', '10') or 'hidden'

    ## bot action
    possible = ['fold']
    if valid_actions[1]['amount'] == 0:
        possible.append('check')
    else:
        possible.append('call')
    if (valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']) != (-1, -1):
        possible.extend(['raise', 'allin'])

    for action in possible:
        df['bot_action'].iloc[-1] = action
        if action == 'call':
            df['action_amt'].iloc[-1] = valid_actions[1]['amount']
        elif action == 'raise':
            raise_bounds = valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']
            raise_amount = min(raise_bounds[0] + 5 * random.randrange(1, 5), raise_bounds[1])
            df['action_amt'].iloc[-1] = raise_amount
        elif action == 'allin':
            df['action_amt'].iloc[-1] = valid_actions[2]['amount']['max']
        else:
            df['action_amt'].iloc[-1] = 0
        df.loc[len(df)] = df.iloc[-1].copy()

    df = df.iloc[:-1].copy()

    ## seats resample
    def seat_iter(init_seat, max_num):
        init_seat = int(init_seat)
        yield from range(init_seat + 1, max_num + 1)
        yield from range(1, init_seat)

    df2 = df.copy()
    for row in df2.index:
        i = 0
        for seat in seat_iter(df2.at[row, 'bot_seat'], max_players):
            i += 1
            df2.loc[row, f"money_{i}"] = df2.at[row, f"s{seat}_money"]

    df3 = df2.drop(
        axis='columns',
        labels=[col for col in df2.columns if '_money' in col and 'bot' not in col] + ['bot_seat']
    )

    df4 = df3.copy()
    df4[[col for col in df2.columns if 'money_' in col]] = df3[[col for col in df2.columns if 'money_' in col]].fillna(0)

    df = df4.copy()

    ## cards resample
    df.replace({'preflop': 'pre'}, inplace=True)

    def parse_cards(s: str) -> tuple:
        carddeck = ['hidden'] + [str(i + 1) for i in range(10)] + ['J', 'Q', 'K', 'A']
        res = []
        for c in s.split():
            if c == 'hidden':
                res.append(('hidden', 'hidden'))
                continue
            res.append((c[:-1], c[-1].lower(),))
        res.sort(key=lambda x: x[1])
        res.sort(key=lambda x: carddeck.index(x[0]), reverse=True)
        while len(res) < 5:
            res.append(('hidden', 'hidden'))
        return tuple(x for t in res for x in t)

    df2 = df.copy()
    board_cols = [f for i in range(1, 6) for f in (f'board_{i}r', f'board_{i}s')]

    df3 = df2.join(
        pd.DataFrame.from_records(
            data=[x for x in df2.board.apply(parse_cards)],
            columns=board_cols,
            index=df2.index
        )
    )

    hand_cols = [f for i in range(1, 3) for f in (f'hand_{i}r', f'hand_{i}s')]
    df4 = df3.join(
        pd.DataFrame.from_records(
            data=[x for x in map(lambda x: x[:4], df3.bot_hand.transform(parse_cards))],
            columns=hand_cols,
            index=df3.index
        )
    )

    df5 = df4.drop(columns=['board', 'bot_hand'])

    df = df5.copy()

    ## rel money
    money_cols = ['bank', 'bet', 'bot_money', 'action_amt'] + [x for x in df.columns if x.startswith('money_')]
    df2 = df.copy()
    df2[money_cols] = df[money_cols].apply(lambda s: s.div(s.bot_money), axis=1)
    df2.drop(columns='bot_money', inplace=True)

    ## Suits diversity
    df4 = df2.copy()
    power_of_suits = lambda s: len(
        {x for x in s[['board_1s', 'board_2s', 'board_3s', 'board_4s', 'board_5s', 'hand_1s', 'hand_2s']] if
         x != 'hidden'})
    df4['suits_amt'] = df2.apply(power_of_suits, axis=1)

    ## High card
    carddeck = ['hidden'] + [str(i + 1) for i in range(10)] + ['J', 'Q', 'K', 'A']
    df5 = df4.copy()
    max_card = lambda s: max(
        s[['board_1r', 'board_2r', 'board_3r', 'board_4r', 'board_5r', 'hand_1r', 'hand_2r']].values,
        key=lambda x: carddeck.index(x),
    )
    df5['high_card'] = df4.apply(max_card, axis=1)

    ## Rank diversity
    df6 = df5.copy()
    ranks = lambda s: [x for x in s[['board_1r', 'board_2r', 'board_3r', 'board_4r', 'board_5r', 'hand_1r', 'hand_2r']]
                       if x != 'hidden']
    df6['rank_div'] = df5.apply(lambda s: len(ranks(s)) / len(set(ranks(s))), axis=1)

    ## Ranks in row
    def get_rank_disp(ser: pd.Series):
        all_ranks = sorted(
            np.delete(
                ser[['board_1r', 'board_2r', 'board_3r', 'board_4r', 'board_5r', 'hand_1r', 'hand_2r']].values,
                np.where(ser[['board_1r', 'board_2r', 'board_3r', 'board_4r', 'board_5r', 'hand_1r',
                              'hand_2r']].values == 'hidden')
            ),
            key=lambda x: carddeck.index(x)
        )

        return (carddeck.index(all_ranks[-1]) - carddeck.index(all_ranks[0])) / 10

    df7 = df6.copy()
    df7['rank_disp'] = df6.apply(get_rank_disp, axis=1)

    df = df7.drop(columns=['Game_ID', 'bot_wins'])

    ## Std Scaler
    # num_feat = [x for x in df.columns if x.startswith('money_')] + ['num_active_pl', 'bank', 'bet', 'action_amt']
    # std_scaler = joblib.load('data/stds5.pkl')
    #
    # scaled = std_scaler.transform(df[num_feat])
    #
    # num_title = [f"{title}_std" for title in num_feat]
    #
    # df2 = df.copy()
    # df2[num_title] = scaled
    # df2.drop(columns=num_feat, inplace=True)

    ## ohe
    df2 = df.copy()
    ohe = joblib.load('data/ohe5.pkl')

    cat_feat = ohe.feature_names_in_

    ohe_cat = ohe.transform(df2[cat_feat])
    df3 = df2.join(
        pd.DataFrame(
            data=ohe_cat,
            columns=ohe.get_feature_names_out(),
            index=df2.index
        )
    ).drop(columns=cat_feat)

    return df3.copy()


def predict(df, strategy: dict) -> tuple:  # action, amount
    # df2 = df.drop(columns=['action_amt'])
    df2 = df.copy()
    model = meta['model']
    possible = []
    for _, s in df2.iterrows():
        possible.append(
            int(s['bot_action_fold'] == 1) * 'fold' +
            int(s['bot_action_check'] == 1) * 'check' +
            int(s['bot_action_call'] == 1) * 'call' +
            int(s['bot_action_raise'] == 1) * 'raise' +
            int(s['bot_action_allin'] == 1) * 'allin'
        )
    pred = dict(zip(possible, model.predict(df2)))
    if 'fold' in pred:
        pred['fold'] = 1
    if 'check' in pred:
        pred['check'] = 1
    if 'call' in pred and pred['call'] == 0:
        pred['call'] == 1 if random.random() < strategy['call'] else 0

    i = -1
    for act, win in pred.items():
        i += 1
        if win == 1:
            if act == 'allin' and random.random() > strategy['allin']:
                continue
            elif act == 'raise' and random.random() > strategy['raise']:
                continue
            action = act
            amount = df.action_amt[i]
    if action == 'allin':
        action = 'raise'
    if action == 'check':
        action = 'call'

    return action, amount


if __name__ == '__main__':
    data = {
        'street': 'preflop',
        'pot': {'main': {'amount': 15}, 'side': []},
        'community_card': ['S4', 'DQ', 'D2'],
        'dealer_btn': 0,
        'next_player': 0,
        'small_blind_pos': 1,
        'big_blind_pos': 2,
        'round_count': 1,
        'small_blind_amount': 5,
        'seats': [
            {'name': 'p1', 'uuid': 'uxfoakjmaauygrgmilntjc', 'stack': 100, 'state': 'participating'},
            {'name': 'p2', 'uuid': 'aowxlipsresmeeyfgsdlwc', 'stack': 95, 'state': 'participating'},
            {'name': 'p3', 'uuid': 'uiszoizblcecccqzcyyspl', 'stack': 90, 'state': 'participating'}
        ],
        'action_histories': {
            'preflop': [{'action': 'SMALLBLIND', 'amount': 5, 'add_amount': 5, 'uuid': 'aowxlipsresmeeyfgsdlwc'},
                        {'action': 'BIGBLIND', 'amount': 10, 'add_amount': 5, 'uuid': 'uiszoizblcecccqzcyyspl'}]}
    }
    bot_uuid = 'uxfoakjmaauygrgmilntjc'
    bot_hand = ['S2', 'D5']
    valid_actions = [
        {'action': 'fold', 'amount': 0},
        {'action': 'call', 'amount': 0},
        {'action': 'raise', 'amount': {'min': 15, 'max': 100}}
    ]

    action_, amount_ = predict(
            parse_from_data(data, bot_uuid, bot_hand, valid_actions),
            {'raise': 0.5, 'allin': 0.1}
        )
    amount_ *= [d['stack'] for d in data['seats'] if d['uuid'] == bot_uuid][0]
    print(action_, amount_)

