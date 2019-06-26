
from opponent import Opponent
from player import Player, humanPlayer
from qLearner import QLearner
import environment as en
import numpy as np



opp = Opponent()
ply = QLearner()
env = en.Environment(np.random.choice(2))


def playGame(ply, opp, startingPlayer, dialog=False):
    #choose players from ['human', 'defaultOpp', 'opponent', 'qOpp', 'deepQopp']
    #ply is player 0, opp is player 1
    opp.player=1
    reward=0
    
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
            if env.state[-1][0]==8:
                reward = env.callout(player=1, ply=ply, opp=opp)
                print('Player overbet. pips: ', env.state[-1][1])
        
        _dialog()
        
        if reward:
            return reward
        
    if str(ply)=='human':
        print('opponent hand was: ', opp.dice)





def qlearning(env, ply, opp, alpha, gamma=1, episodes=10):
    for episode in range(episodes):
        
        
        print(f"Episode: {episode}")
        #env.__init__(np.random.choice(2)) #reset env with random starting player
        env.__init__(1) #opponent always starts, otherwise we would have to train seperate q-matrices for the first round with no info
        
        ply.reset()
        opp.reset()
        
        reward = playGame(ply, opp, startingPlayer=env.beginner, dialog=True)
        
        
        ply.qCallout, ply.qNumsToBet, updatePtb, qPtb, ply.qBluff, ply.qRebluff = ply.train(env, reward, alpha=alpha)
        if updatePtb: ply.qPipsToBet = qPtb
        
        



def showTraining(ply, saves):
    
    print('Number of trained parameters:')
    print('Nums to Bet: ', np.sum(ply.qNumsToBet!=saves[0]))
    print('Callout: ', np.sum(ply.qCallout!=saves[1]))
    print('Bluff: ', np.sum(ply.qBluff!=saves[2]))
    print('Rebluff: ', np.sum(ply.qRebluff!=saves[3]))
    print('Pips to Bet: ', np.sum(ply.qPipsToBet!=saves[4]))




np.random.seed(0)


ply.reset()
opp.reset()
#ply.dice = [5,2,3,3,1]
#opp.dice = [2,2,2,2,4]

saves = (ply.qNumsToBet, ply.qCallout, ply.qBluff, ply.qRebluff, ply.qPipsToBet)

print('Player Dice:', ply.dice)
print('Opponent Dice:', opp.dice)
qlearning(env=env, ply=ply, opp=opp, alpha=0.5, episodes=1)

showTraining(ply, saves)


print(ply.history)


