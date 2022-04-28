
#the base class Player, from which all player classes inherit, and the humanPlayer class for interactive testing

class Player():
    '''other player classes inherit from this one. also has built-in default behaviour, so works as (really bad) player'''
    def __init__(self):
        import numpy as np
        self.dice = np.random.choice(range(1,7),5)
        self.counts = [list(self.dice).count(i) for i in range(1,7)]
        self.player = 0

        #important note: counts[0] will display number of two's in the end
        #ones are jokers and cannot be bet so we kick them out for simplicity of state space
        if self.counts.count(0)==1:
            #special rule: all different dice = 0 of all
            self.counts = [0,0,0,0,0]

        else:
            #special rule: ones count as joker (added to all others)
            self.counts = [self.counts[i]+self.counts[0] for i in range(1,6)]
            #special rule: having 5 of one counts as 6 instead
            self.counts = [6 if i==5 else i for i in self.counts]

    def __str__(self):
        '''override this method'''
        #method for returning name of player
        return 'default'

    def calling(self, env):
        '''override this method'''
        #method for checking, whether to callout
        return False

    def betting(self, env):
        '''override this method'''
        #method for deciding a bet when not calling out
        currNums, currPips = env.state[-1]
        newNums, newPips = [currNums+1, 2] if currPips==6 else [currNums, currPips+1]

        return [newNums, newPips]

    def makeTurn(self, env):
        #check whether to callout and what to bet
        calling = self.calling(env) if env.turn >= 1 else False
        num, pips = [100,6] if calling else self.betting(env)

        return [calling, [num, pips]]

    def reset(self):
        self.__init__()

    def end(self, reward):
        print('Player got reward: ', reward)


class humanPlayer(Player):
    '''this class can be used to play the game yourself with terminal inputs'''

    def __str__(self):
        return 'human'

    def calling(self, env):

        if env.turn >= 1:
            print('your dice: ', self.dice)
            print('opponent bet: ', env.state[-1])
            callInput = input('Callout? (y/n)')
            calling = True if callInput == 'y' else False
        else:
            calling = False

        return calling

    def betting(self, env):


        print('your dice: ', self.dice)

        num = int(input('How many?'))
        pips = int(input('Of which?'))

        print('Your bet: ', num, pips)
        return [num, pips]









