


h = humanPlayer()

ply.eps=0



def human_game(ply, opp, dialog=True):
    #choose players from ['human', 'defaultOpp', 'opponent', 'qOpp', 'deepQopp']
    #ply is player 0, opp is player 1
    env = Environment(np.random.choice(2))
    ply.player=0
    opp.player=1
    reward=0
    startingPlayer = env.beginner

    print('opponent hand was: ', opp.dice)

    def _dialog(ply='Player', dialog=dialog):
        if dialog:
            if reward: print('Reward: ', reward)
            else: print(ply, 'bet: ', env.state[-1])

    if startingPlayer==0:
        called, bet = ply.makeTurn(env)
        env.finishTurn(bet)

        _dialog()

    while True:
        # for debugging insert: print('current state :', env.state)

        #opponent turn:
        called, bet = opp.makeTurn(env)

        if called:
            reward = env.callout(player=1, ply=ply, opp=opp)
        else:
            env.finishTurn(bet)
            if env.state[-1][0]==8:
                reward = env.callout(player=0, ply=ply, opp=opp)
                if dialog:
                    print('Opponent overbet. pips: ', env.state[-1][1])

        _dialog(ply='Opponent')

        if reward:
            return reward

        #player turn:
        called, bet = ply.makeTurn(env)

        if called:
            reward = env.callout(player=0, ply=ply, opp=opp)
        else:
            env.finishTurn(bet)
            if env.state[-1][0]>=8:
                reward = env.callout(player=1, ply=ply, opp=opp)
                if dialog:
                    print('Player overbet. pips: ', env.state[-1][1])

        _dialog()

        if reward:
            return reward

    if str(ply)=='human':
        print('opponent hand was: ', opp.dice)


h.reset()
ply.reset()
human_game(h, ply)

