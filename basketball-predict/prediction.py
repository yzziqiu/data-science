import pandas as pd
import numpy as np
import math
import csv
import random
from sklearn import cross_validation, linear_model

# intialize elo class
base_elo = 1600
team_elos = {}
team_stats = {}
x = []
y = []
folder = 'data'

# initialize Miscellaneous, opponent, team from csv files
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'],axis=1)
    new_Ostat = Ostat.drop(['Rk','G','MP'],axis=1)
    new_Tstat = Tstat.drop(['Rk','G','MP'],axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1,new_Tstat,how='left',on='Team')
    return team_stats1.set_index('Team', inplace = False, drop = True)

def get_elo(team):
    try:
        return team_elos[team]
    except:
        # no elo, given elo_base
        team_elos[team] = base_elo
        return team_elos[team]

# team elo
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) /400
    odds = 1 / (1+math.pow(10,exp))
    # modify k value
    if winner_rank <2100:
        k=32
    elif winner_rank >=2100 and winner_rank <2400:
        k=24
    else:
        k=16
    new_winner_rank = round(winner_rank+k*((1-odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff
    return new_winner_rank, new_loser_rank

def build_dataSet(all_data):
    print("building dataset...")
    x = []
    skip = 0
    for index, row in all_data.iterrows():
        Wteam = row['WTeam']
        Lteam = row['LTeam']
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)
        # host adding 100 to elo
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        team1_features = [team1_elo]
        team2_features = [team2_elo]
        # adding info from data
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
            for key,value in team_stats.loc[Lteam].iteritems():
                team2_features.append(value)
        # seperate the data left and right
        if random.random() > 0.5:
            x.append(team1_features + team2_features)
            y.append(0)
        else:
            x.append(team2_features + team1_features)
            y.append(1)
            if skip == 0:
                print x
            skip = 1

# update elo value
    new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
    team_elos[Wteam] = new_winner_rank
    team_elos[Lteam] = new_loser_rank
    return np.nan_to_num(x), y

# logistic regression
if __name__ == "__main__":
    Mstat = pd.read_csv(folder + '/Miscellaneous_Stats.csv')
    Ostat = pd.read_csv(folder + '/Opponent_Per_Game_Stats.csv')
    Tstat = pd.read_csv(folder + '/Team_Per_Game_Stats.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)
    result_data = pd.read_csv(folder + '2016-2017results.csv')
    x, y = build_dataSet(result_data)

    # training
    print("fitting on %d game samples" % len(x))
    model = linear_model.LogisticRegression()
    model.fit(x,y)
    print("doing cross_validation")
    print(cross_validation.cross_val_score(model, x, y,cv=10,scoring='accuracy',n_jobs=-1).mean())

    def predict_winner(team_1,team_2,model):
        features = []
        features.append(get_elo(team_1)) #geust
        for key, value in team_stats.loc[team_1].iteritems():
            features.append(value)
            features.append(get_elo(team_2)+100)
            for key,value in team_stats.loc[team_2].iteritems():
                features.append(value)
        features = np.nan_to_num(features)
        return model.predict_proba([features])

        print('Predicting on new schedule..')
    schedule1718 = pd.read_csv(folder + '/17-18Schedule.csv')
    result = []
    for index, row in schedule1718.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])
            with open('17-18Result.csv', 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['win', 'lose', 'probability'])
                writer.writerows(result)
