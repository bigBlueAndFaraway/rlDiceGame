
'''environment'''

class Environment():
    def __init__(self, beginner):
        self.state = [[2,6]]
        self.turn = 0
        self.beginner = beginner #=1 for opponent, =0 for player
        

    def finishTurn(self, betDice):
        #call this method at the end of a turn to add last bet
        
        self.state += [betDice] #state is simply list of all bets
        self.turn += 1
        
        
    
    def callout(self, player, ply, opp):
        betNum, betPips = self.state[-1]
        sumNum = opp.counts[betPips-2] + ply.counts[betPips-2] #-2 because index starts at 0 and we threw out ones in self.counts
        
        winner = player if sumNum < betNum else 1-player
    
        reward = 1 if winner==0 else -1
        
        return reward
    
