
import numpy as np
from player import Player

'''hard coded player'''

class Opponent(Player):
    
    def __str__(self):
        return 'hardcoded'
    
    
    def lowestAllowedNum(self, env, pips):
        if len(env.state)>1:
            lastNum, lastPips = env.state[-1]
        else: lastNum, lastPips = 0,0
        
        if lastPips < pips:
            return lastNum
        else: return lastNum+1
    
    def betting(self, env):
        #bluffing works in this game for two reasons: firstly, if i honestly bet my best pips, the opponent will know my hand and could f.e. bet my pips back
        #secondly by bluffing i can hope he honestly bets the same pips, then i can easily call him out
        bluffChance = 1/3 * 1/(env.turn+1)
        bluff = np.random.choice([0,1], p=[1-bluffChance, bluffChance])
        
        if bluff:
            if len(env.state)>0:
                pips = env.state[-1][1]
                num = max(env.state[-1][0], self.counts[pips-2])+1
            else:
                pips = np.random.choice([i for i in range(5) if self.counts[i]==np.min(self.counts)]) #range(5): ones (at index 0) have been kicked out
                num = self.counts[pips-2] +  np.random.choice([0,1,2])
        else:
            pips = np.argmax(self.counts)+2 #bet pips we have most of
            num = max(self.lowestAllowedNum(env, pips), self.counts[pips-2]) + np.random.choice([0,1,2])
        
        return [min(8, num), pips]
        
    def checkCallout(self, env):
        lastNum, lastPips = env.state[-1]
        coProb = min((2**max(lastNum - self.counts[lastPips-2], 0)) / (2**5), 1)
        callout = np.random.choice([0,1], p=[1-coProb, coProb])
        
        return bool(callout)
    
    def makeTurn(self, env):
        if env.turn>1:
            callout = self.checkCallout(env)
        else: callout = False
        if callout:
            return [True, [0,0]]
        else:
            return [False, self.betting(env)]
    

