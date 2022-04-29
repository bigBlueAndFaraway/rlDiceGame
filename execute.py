
from opponent import Opponent
from player import Player, humanPlayer
from qLearner import QLearner
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import copy



def play_game(ply, opp, env, dialog=False):
    #choose players from ['human', 'defaultOpp', 'opponent', 'qOpp', 'deepQopp']
    #ply is player 0, opp is player 1

    ply.player=0
    opp.player=1
    reward=0
    startingPlayer = env.beginner

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



def qlearning(env, ply, opp, alpha, gamma=1, episodes=10, last_reward=0):

    rew_hist=[last_reward]
    for episode in range(episodes):

        #if episode in range(0,100000, 1000):
        #    print(f"Episode: {episode}")
        #env.__init__(np.random.choice(2)) #reset env with random starting player
        env = Environment(np.random.choice(2)) #opponent always starts, otherwise we would have to train seperate q-matrices for the first round with no info

        ply.reset()
        opp.reset()

        reward = play_game(ply, opp, env=env, dialog=False)
        #rew_hist += [reward + rew_hist[-1]]
        rew_hist += [reward]

        # saves = (ply.qNumsToBet, ply.qCallout, ply.qBluff, ply.qRebluff, ply.qPipsToBet)

        '''
        vec = ply.train(env, reward, alpha=alpha)
        if vec:
            ply.qCallout, ply.qNumsToBet, updatePtb, qPtb, ply.qBluff, ply.qRebluff = vec
            if updatePtb: ply.qPipsToBet = qPtb
        else: print('vec empty')
        '''

        ply.train(env, reward, alpha=alpha)
        #if str(opp)=='qLearner':
        #    opp.train(env, -reward, alpha=alpha)

    return rew_hist, ply, opp

def test_play(env, ply, opp, episodes=10000):
    #every 10k rounds or so we check how well ply performs. if it wins > 55%, we copy it into opp before continuing the training
    rew_hist=[]

    e = copy.deepcopy(ply.eps)
    ply.eps = 0

    for episode in range(episodes):
        env = Environment(np.random.choice(2)) #opponent always starts, otherwise we would have to train seperate q-matrices for the first round with no info

        ply.reset()
        opp.reset()

        reward = play_game(ply, opp, env=env, dialog=False)
        #rew_hist += [reward + rew_hist[-1]]
        rew_hist += [reward]

    ply.eps = e

    return rew_hist


def showTraining(ply, saves):

    print('Number of trained parameters:')
    print('Nums to Bet: ', np.sum(ply.qNumsToBet!=saves[0]))
    print('Callout: ', np.sum(ply.qCallout!=saves[1]))
    print('Bluff: ', np.sum(ply.qBluff!=saves[2]))
    print('Rebluff: ', np.sum(ply.qRebluff!=saves[3]))
    print('Pips to Bet: ', np.sum(ply.qPipsToBet!=saves[4]))

def showDecisions(ply):
    rebluffs = [i[0] for i in ply.decisionHistory]
    bluffs = [i[1] for i in ply.decisionHistory]

    print('Rebluffs: ')
    plt.plot([np.mean(rebluffs[i*10000:(i+1)*10000]) for i in range(round(len(rebluffs)/10000))], 'b-')
    print('Bluffs: ')
    plt.plot([np.mean(bluffs[i*10000:(i+1)*10000]) for i in range(round(len(bluffs)/10000))], 'r-')
    #print('Honest Plays: ', len(ply.decisionHistory)-bluffs-rebluffs)


def analysis(ply):
    print('callout value:', np.sum(ply.qCallout)/np.size(ply.qCallout))
    print('bluff value:', np.sum(ply.qBluff)/np.size(ply.qBluff))
    print('rebluff value:', np.sum(ply.qRebluff)/np.size(ply.qRebluff))



if False:# or __name__ == 'main':
    opp = Opponent()
    ply = QLearner()
    env = Environment(np.random.choice(2))
    #play_game(ply=ply, opp=opp, env=env)


#saves = (ply.qNumsToBet, ply.qCallout, ply.qBluff, ply.qRebluff, ply.qPipsToBet)

hist = list(np.repeat(0,100))
callout_hist=[np.sum(ply.qCallout)/np.size(ply.qCallout)]
opp_hist = list(np.repeat(0,100))
#opp_callout_hist=[np.sum(opp.qCallout)/np.size(opp.qCallout)]



def load_players():
    env = Environment(np.random.choice(2))
    ply = pickle.load(open('C:\\Users\\Lennart\\Desktop\\qPlayer.pickle', 'rb'))
    ply.eps = .2
    opp = copy.deepcopy(ply)
    opp.eps = 0

    return ply, opp, env

def save_player(path='C:\\Users\\Lennart\\Desktop\\qPlayerNew.pickle'):
    pickle.dump(ply, open(path, 'wb'))


def run_training(ply, opp, env, start_round, end_round, episodes, eps=.2, alpha=.99, auto_update=False):

    import pyttsx3
    sound_engine = pyttsx3.init()
    sound_engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')


    x = 0
    test_results = []
    ply.eps=eps
    opp.eps=0

    #ply = copy.deepcopy(opp)
    #ply.eps = .3

    sound_engine.say('training initialized')
    sound_engine.runAndWait()

    start = time.time()
    for i in range(start_round, end_round):

        ply.decisionHistory = []
        ply.history = ply.history[-100:]
        #clean history to prevent large model size. remove to analyse decision history


        epstart = time.time()
        print('__________Round: ', i)
        new_hist, ply, opp = qlearning(env=env, ply=ply, opp=opp, alpha=alpha**i, episodes=episodes, last_reward=hist[-1])
        hist += new_hist

        #callout_hist += [np.sum(ply.qCallout)/np.size(ply.qCallout)]

        if i%10 == 0:
            test = test_play(env, ply, opp)
            print('TEST SCORE: ', np.mean(test))
            test_results += [np.mean(test)]
            if auto_update and np.mean(test) > .05:
                #this func copies ply to opp as soon as he got significantly better
                opp = copy.deepcopy(ply)
                opp.eps=0
                x+=1


        print("Ep Time: ", time.time()-epstart)




    #test = test_play(env, ply, opp)
    #plt.plot([np.mean(test[i*1000:(i+1)*1000]) for i in range(10)])

    end = time.time()
    print(end-start)

    sound_engine.say('training complete')
    sound_engine.runAndWait()

    plt.plot(test_results)

run_training(ply, opp, env, 150, 500, 30000)



if False:
    #non_zero_hist = [i for i in hist if i!=0]

    mov_avg = pd.Series([np.mean(hist[i*200000 : (i+1)*200000]) for i in range(round(len(hist)/200000 - 1))])


    plt.plot(mov_avg, 'g-')

if False:
    opp_mov_avg = [sum([min(i,0) for i in opp_hist[-len(opp_hist)+j:-len(opp_hist)+1000+j]])/1000 for j in range(len(opp_hist)-1000)]
    plt.plot(opp_mov_avg)





#winner_hist_mov_avg = [np.mean(winner_hist[i*1000:(i+1)*1000]) for i in range(5000)]
#plt.plot(winner_hist_mov_avg)

#pickle.dump(opp, pickle_out_opp)



#ply = pickle.load(pickle_in)
#opp = pickle.load(pickle_in_opp)

#analysis(ply)

#showDecisions(ply)



#a=ply.decisionHistory
#b=[i[1] for i in a]
#c = [np.mean(b[i*1000:(i+1)*1000]) for i in range(87)]

#plt.plot(c)

#winner_hist_avg = [np.mean(winner_hist[i*1000:(i+1)*1000]) for i in range(50)]
#plt.plot(winner_hist_avg)

#showTraining(ply, saves)








'''
reward = playGame(ply, opp, startingPlayer=env.beginner, dialog=False)
ply.history

numTurns = len(env.state)
actionHistory = env.state[1+env.beginner: numTurns-1+env.beginner:2]
numPlayerTurns = len(actionHistory)



env.state


    def findNumsToBet(self, qNumsToBet, betPips, betNums, pipsToBet, prevBetPips, pbnIsRelevant):
        s = (int(betPips==pipsToBet), betNums-2, self._numIndex(self.counts[pipsToBet-2]), int(prevBetPips==betPips&pbnIsRelevant))
        add = 0 if betPips < pipsToBet else 1
        allowedNums = np.concatenate((np.repeat(-1, betNums-2+add), self.qNumsToBet[s][betNums-2+add:])) #setting other numbers to -1, which is minimum score
        numsToBet = np.argmax(allowedNums)+2

        self.stateNtb = s

        return numsToBet

ply._numIndex(ply.counts[pipsToBet-2])

betPips = 3
betNums = 4
pipsToBet = 3
allowedNums = np.concatenate((np.repeat(-1, betNums-2+add), self.qNumsToBet[s][betNums-2+add:]))
'''