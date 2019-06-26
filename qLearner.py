
import numpy as np
from player import Player

class QLearner(Player):
    
    def __init__(self):
        
        super().__init__()
        
        self.qCallout = np.random.random((5, 2, 6, 5, 2))
            #max(0, min(4, betNums-counts[betPips])), prevBetPips==betPips&pbnIsRelevant, counts[betPips], max(0,min(4, numToBet-counts[pipsToBet])) /callout
            #states = 600
    
        self.qPipsToBet = np.random.random((5, 2, 10, 2))
            #betPips, prevBetPips==betPips&pbnIsRelevant, _flattenedBettablePips(bettablePips) /pipsToBet (out of two highest nums available)
            #states = 500
                
        self.qNumsToBet = np.random.random((2, 6, 6, 2, 7))
            #betPips==pipsToBet, betNums, counts[pipsToBet], prevBetPips==betPips&pbnIsRelevant /numsToBet (7 because you can bet 8 with immediate callout)
            #states = 1008
        
        self.qBluff = np.random.random((6, 5, 3, 2))
            #betNum, max(0,min(numToBet - counts[pipsToBet], 4)), min(count[bluffPips], 2) /bluff
            #states = 180
            
        self.qRebluff = np.random.random((6, 5, 2))
            #betNum, max(0,min(numToBet - counts[pipsToBet], 4)) /rebluff
            #states = 60
        
        self.history = []
        self.decisionHistory = []
        self.statePtb = ()
        self.stateNtb = ()
        self.stateCallout = ()
        self.stateBluff = ()
        self.stateRebluff = ()
            #useful for saving information to update q values
            
        self.gamma = 0.9
    
    def reset(self):
        super().__init__()
        self.history = []
        self.decisionHistory = []


    def __str__(self):
        return 'qLearner'
    
    def featureExtraction(self, env):
        #for deep q learning we might replace this function with a neural network

        betNums, betPips = env.state[-1]
        if len(env.state)>2:
            prevBetNums, prevBetPips = env.state[-3]
            consistentBet = prevBetPips==betPips
        else:
            prevBetNums, prevBetPips = [0,np.random.choice(5)+2] #just use 0 of any of the pips for easy usage (will be set as irrelevant later)
            consistentBet = False
        
        pbnIsRelevant = prevBetNums < 3 + np.random.choice(2)
        
        includePrevBetPips = prevBetPips==betPips & pbnIsRelevant
        
        #betNumsFiltered = betNums if betNums > 3 else 3
        #we will just start with two sixes by default that cant be called out
        
        
        def _countMapping(counts, betPips):
            #filtering out (up to) two highest counts to take into account for simplification of state space
            allHigh = np.argwhere(counts==np.max(counts)).flatten().tolist()            
            
            if len(allHigh) > 2 and betPips-2 in allHigh:
                allHigh.remove(betPips-2)
                pipList = [betPips-2, np.random.choice(allHigh)]
            elif len(allHigh) > 2:
                pipList = list(map(int, np.random.choice(allHigh, size=2, replace=False))) #map turns int64 into int
            else: pipList = allHigh
            
            return [pip+2 for pip in pipList]

        bettablePips = _countMapping(self.counts, betPips)
        
        bluffPips = [pip+2 for pip in np.argwhere(self.counts==np.min([i for i in self.counts if i!=betNums])).flatten().tolist()]



        return consistentBet, pbnIsRelevant, includePrevBetPips, bettablePips, betNums, betPips, prevBetNums, prevBetPips, bluffPips

    def calling(self, env, numsToBet, pipsToBet):
        consistentBet, pbnIsRelevant, includePrevBetPips, bettablePips, betNums, betPips, prevBetNums, prevBetPips, bluffPips, _, _\
            = self.completeState
        
        overbetOpp = max(0, min(4, betNums-self.counts[betPips-2]))
        overbetPly = max(0, min(4, numsToBet-self.counts[pipsToBet-2]))
        s = (overbetOpp, int(prevBetPips==betPips&pbnIsRelevant), self._numIndex(self.counts[betPips-2]), overbetPly)
        
        action_callout = np.argmax(self.qCallout[s])
        
        self.stateCallout = s
        
        if env.turn==0:
            return False
        else:
            return action_callout
    
    def findPipsToBet(self, qPipsToBet, betPips, bettablePips, prevBetPips, pbnIsRelevant):
        
        if len(bettablePips)==1:
            trainMe=False
            pipsToBet = bettablePips[0]

            s = (10,10,10) #this state won't be accessed for training because trainMe is set on False. 10s work as sanity check
            
        else:
            flat = self._flattenBettablePips(bettablePips=bettablePips)
            s = (betPips-2, int(prevBetPips==betPips&pbnIsRelevant), flat)
            low, high = min(bettablePips), max(bettablePips)
            highLow = np.argmax(self.qPipsToBet[s])
            pipsToBet = low if highLow==0 else high
            trainMe=True
        
        self.statePtb = s
        
        return pipsToBet, trainMe
    
    def _flattenBettablePips(self, bettablePips):
        #turns list of two pips into integer
        low, high = min(bettablePips), max(bettablePips)
        if low==2:
            return high-3
        elif low==3:
            return high
        elif low==4:
            return high+2
        else:
            return 9
    
    def _unflattenBettablePips(self, flattened):
        #reverses flattening. obviously
        if flattened<4:
            low, high = 2, flattened+3
        elif flattened<7:
            low, high = 3, flattened
        elif flattened <9:
            low, high = 4, flattened-2
        else: low, high = 5, 6
        return [low,high]
    
    def _numIndex(self, nums):
        a = 5 if nums==6 else nums
        return a
    
    def findNumsToBet(self, qNumsToBet, betPips, betNums, pipsToBet, prevBetPips, pbnIsRelevant):
        s = (int(betPips==pipsToBet), betNums-2, self._numIndex(self.counts[pipsToBet-2]), int(prevBetPips==betPips&pbnIsRelevant))
        add = 0 if betPips < pipsToBet else 1
        allowedNums = np.concatenate((np.repeat(-1, betNums-2+add), self.qNumsToBet[s][betNums-2+add:])) #setting other numbers to -1, which is minimum score

        numsToBet = np.argmax(allowedNums)+2
        
        self.stateNtb = s
        
        return numsToBet


    def bluffing(self, qBluff, betNum, betPips, pipsToBet, numToBet, bluffPips):
        
        action_bluffPips = np.random.choice(bluffPips)
        
        
        s = (betNum-2, max(0,min(numToBet-self.counts[pipsToBet-2],4)), min(self.counts[action_bluffPips-2], 2))
        action_bluff = np.argmax(self.qBluff[s])
        
        self.stateBluff = s
        
        return action_bluff, action_bluffPips




    def rebluff(self, qRebluff, betNum, numToBet, pipsToBet, betPips, bluffPips):
        #only execute this method if betPips in bluffPips
        s = (betNum-2, max(0,min(numToBet-self.counts[pipsToBet-2],4)))   
        action_rebluff = np.argmax(self.qRebluff[s])
        
        self.stateRebluff = s
        
        if betPips in bluffPips:
            return action_rebluff
        else: return 0
    
    
    def betting(self, env):
        
        consistentBet, pbnIsRelevant, includePrevBetPips, bettablePips, betNums, betPips, prevBetNums, prevBetPips, bluffPips\
            = self.completeState
        
        pipsToBet, trainMe_ptb = self.findPipsToBet(self.qPipsToBet, betPips, bettablePips, prevBetPips, pbnIsRelevant)
        numsToBet = self.findNumsToBet(self.qNumsToBet, betPips, betNums, pipsToBet, prevBetPips, pbnIsRelevant)
        action_bluff, action_bluffPips = self.bluffing(self.qBluff, betNums, betPips, pipsToBet, numsToBet, bluffPips)
        action_rebluff = self.rebluff(self.qRebluff, betNums, numsToBet, pipsToBet, betPips, bluffPips)
        
        
        if action_rebluff:
            newPips = betPips
            newNums = betNums+1
        elif action_bluff:
            newPips = action_bluffPips
            newNums = betNums+1
        else:
            newPips, newNums = pipsToBet, numsToBet
        
        self.completeState += [action_rebluff, action_bluff]
        self.decisionHistory += [[action_rebluff, action_bluff]]
        self.history += [[self.stateNtb, self.statePtb, self.stateCallout, self.stateBluff, self.stateRebluff, trainMe_ptb]]
        
        return [newNums, newPips]
    
    def makeTurn(self, env):
        #overriding original func because calling requires nums
        self.completeState = list(self.featureExtraction(env))
        
        #check whether to callout and what to bet
        nums, pips = self.betting(env)
        calling = self.calling(env, nums, pips)
        if calling:
            return [True, [100,6]]
        else:
            return [False, [nums, pips]]
        
    

    def train(self, env, reward, alpha):
        '''updating the five q matrices at the end of episode working backwards'''
        gamma=self.gamma
        
        print('training')

        def _updateEpisode(env, reward=reward, alpha=alpha, gamma=gamma):
            numTurns = len(env.state)
            actionHistory = env.state[1+env.beginner: numTurns-1+env.beginner:2] #every second action was by the player
            numPlayerTurns = len(actionHistory)
            
            bettablePips = self.completeState[4]
            
            for turn in range(numPlayerTurns):
                state = self.history[-turn-1]
                action = actionHistory[-turn-1]
                
                if turn!=0:
                    nextAction = actionHistory[-turn]
                    nextState = self.history[-turn]
                else:
                    nextAction = [None, None]
                    nextState = [None, None, None, None, None]       


                action_rebluff, action_bluff = self.decisionHistory[-turn]
                
                currReward = reward if turn == 0 else 0
                if not currReward: nextAction_rebluff, nextAction_bluff = self.decisionHistory[-turn-1]
                else: nextAction_rebluff, nextAction_bluff = None, None 
                
                qCallout, qNumsToBet, updatePtb, qPipsToBet, qBluff, qRebluff = self.qCallout.copy(), self.qNumsToBet.copy(), False, self.qPipsToBet.copy(), self.qBluff.copy(), self.qRebluff.copy()
                
                if currReward:
                    qCallout =_updateValue(Q=self.qCallout, state=state[2], action=state[-1], reward=currReward)
                    if not state[-1]: #i.e. if we didn't call out
                        
                        indAction = self._numIndex(action[0])-3
                        qNumsToBet = _updateValue(Q=self.qNumsToBet, state=state[0], action=indAction, reward=currReward)
                        if state[5]:
                            #i.e. if we had several pips to choose from
                            qPipsToBet = _updateValue(Q=self.qPipsToBet, state=state[1], action=bettablePips.index(action[1]), reward=currReward)
                            updatePtb = True
                        qBluff = _updateValue(Q=self.qBluff, state=state[3], action=action_bluff, reward=currReward)
                        qRebluff = _updateValue(Q=self.qRebluff, state=state[4], action=action_rebluff, reward=currReward)

                else:
                    if turn!=numPlayerTurns-1 and env.beginner==0:
                        #dont update qCallout in first round if player started, because we have no info
                        qCallout = _updateValue(Q=self.qCallout, state=state[2], action=0, nextState=nextState[2], nextAction=int(nextState[-1]))
                    indAction = self._numIndex(action[0])-3
                    indNextAction = self._numIndex(nextAction[0])-3
                    qNumsToBet = _updateValue(Q=self.qNumsToBet, state=state[0], nextState=nextState[0], action=indAction, nextAction=indNextAction)
                    if state[5]:
                        qPipsToBet = _updateValue(Q=self.qPipsToBet, state=state[1], nextState=nextState[1], action=bettablePips.index(action[1]), nextAction=bettablePips.index(nextAction[1]))
                        updatePtb=True
                    qBluff = _updateValue(Q=self.qBluff, state=state[3], nextState=nextState[3], action=action_bluff, nextAction=nextAction_bluff)
                    qRebluff = _updateValue(Q=self.qRebluff, state=state[4], nextState=nextState[4], action=action_rebluff, nextAction=nextAction_rebluff)
                
                return qCallout, qNumsToBet, updatePtb, qPipsToBet, qBluff, qRebluff


        def _updateValue(Q, state, action, nextState=None, nextAction=None, alpha=alpha, gamma=gamma, reward=0):
            saveQ = Q.copy()
            
            print('STATE: ', state)
            print('ACTION: ', action)
            print('NEXTACTION: ', nextAction)
            print('NEXTSTATE: ', nextState)
            
            if reward:
                Q[state][action] += alpha * ( reward  - Q[state][action] )
                
                print('STATE: ', state)
                print('ACTION: ', action)
                #print('DIM: ', Q[state][action])
                #print('Q changed by ', alpha * ( reward  - Q[state][action] ))
            else:
                Q[state][action] += alpha * ( gamma * Q[nextState][nextAction] - Q[state][action] )
                
                print('STATE: ', state)
                print('ACTION: ', action)
                #print('DIM here: ', Q[state][action])
                #print('Q changed by ', alpha * ( gamma * Q[nextState][nextAction] - Q[state][action] ))
                
            if (Q==saveQ).all():
                print('SOMETHING WENT WRONG')
            else:
                print('NICE')
            return Q
        
        '''
        qCallout, qNumsToBet, updatePtb, qPtb, qBluff, qRebluff = _updateEpisode(env)
        
        return qCallout, qNumsToBet, updatePtb, qPtb, qBluff, qRebluff
        '''
        return _updateEpisode(env)


# =============================================================================
# 
# print(ply.qCallout[ply.history[-1][2]][int(ply.history[-1][-1])])
# 
# 
# 
# print(ply.history[-1][-1])
# 
# print()
# =============================================================================





