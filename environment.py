
'''environment'''

class Environment():
    def __init__(self, beginner):
        self.state = [[2,6]]
        self.turn = 0
        self.beginner = beginner #=1 for opponent, =0 for player
        self.caller = None
        self.reward = None
        self.winner = None

    def reset(self, beginner):
        self.__init__(beginner)

    def finishTurn(self, betDice):
        #call this method at the end of a turn to add last bet

        self.state += [betDice] #state is simply list of all bets
        self.turn += 1



    def callout(self, player, ply, opp):
        self.caller = player
        betNum, betPips = self.state[-1]
        sumNum = opp.counts[betPips-2] + ply.counts[betPips-2] #-2 because index starts at 0 and we threw out ones in self.counts

        self.winner = player if sumNum < betNum else 1-player

        self.reward = 1 if self.winner==0 else -1

        #if str(opp) == 'human':
        #    if self.reward==1:
        #        print('you lost')
        #    else:
        #        print('you won')

        return self.reward

