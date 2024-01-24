import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

data = pd.read_csv('./data/ufcData.csv')
print(data.head())
print(data.shape)

# Keeping the columns needed for calculations
filteredData = data[[
    'R_fighter',
    'B_fighter',
    'R_odds',
    'B_odds',
    'Winner',
    'B_current_lose_streak',
    'B_current_win_streak',
    'B_losses',
    'B_total_rounds_fought',
    'B_wins',
    'R_current_lose_streak',
    'R_current_win_streak',
    'R_losses',
    'R_total_rounds_fought',
    'R_wins',
    'B_match_weightclass_rank',
    'R_match_weightclass_rank',
]]

print(filteredData.head())
print(filteredData.shape)

# In data, if the fighter has no rank its shown as NaN, we'll replace it with -1 to recognize it in later calculations
filteredData = filteredData.fillna(-1)

# Iteration trough the every fight(row) in the dataset

for index, row in filteredData.iterrows():
  
  # Calculation are written so that favorable outcome for red fighter is positive and for blue fighter is negative

  # Assuming that having more rounds(=experience) is better
  filteredData.loc[index, 'dTotal_rounds_fought'] = row['R_total_rounds_fought'] - row['B_total_rounds_fought'] 
  
  # Assuming that having more wins is better
  filteredData.loc[index, 'dWins'] = row['R_wins'] - row['B_wins']

  # Assuming that having more losses is worse
  filteredData.loc[index, 'dLosses'] = row['B_losses'] - row['R_losses']
  
  # If both have win streak we calculate the difference between win streaks
  if filteredData['B_current_win_streak'][index] and filteredData['R_current_win_streak'][index]:
    filteredData.loc[index, 'dWin_streak'] = row['R_current_win_streak'] - row['B_current_win_streak']
  
  # If red corner doesn't have a win streak we'll count it's lose streak in favor of the blue corner
  elif filteredData['B_current_win_streak'][index] and not filteredData['R_current_win_streak'][index]:
    filteredData.loc[index, 'dWin_streak'] = -row['R_current_lose_streak'] - row['B_current_win_streak']

  # If blue corner doesn't have a win streak we'll count it's lose streak in favor of the red corner
  elif not filteredData['B_current_win_streak'][index] and filteredData['R_current_win_streak'][index]:
    filteredData.loc[index, 'dWin_streak'] = row['R_current_win_streak']+row['B_current_lose_streak']
  
  # If both have lose streaks we'll subtract the red corner's lose streak from the blue corner's lose streak
  # -> we end up >0 difference if red corner has shorter lose streak and <0 if blue corner has shorter lose streak
  else:
    filteredData.loc[index, 'dWin_streak'] = row['B_current_lose_streak'] - row['R_current_lose_streak']

  # If both have ranking we calculate the difference between rankings (0 is the best rank, 15 worst ->  diff maps from -15 to 15)
  if filteredData['B_match_weightclass_rank'][index] and filteredData['R_match_weightclass_rank'][index]:
    filteredData.loc[index, 'dRanking'] = row['B_match_weightclass_rank'] - row['R_match_weightclass_rank']

  # For non ranked fighters we'll assume the difference to 0 
  else:
    filteredData.loc[index, 'dRanking'] = 0


print(filteredData.head())
print(filteredData.shape)

# Keeping only the columns we need 
finalData = data[[
    'R_fighter',
    'B_fighter',
    'R_odds',
    'B_odds',
    'Winner',
    'dTotal_rounds_fought',
    'dWins',
    'dLosses',
    'dWin_streak',
    'dRanking'
]]

print(finalData.head())
print(finalData.shape)
print('-----------------------------------'*4)


X = finalData[['dTotal_rounds_fought', 'dWins', 'dLosses', 'dWin_streak', 'dRanking']]
y = finalData['R_odds']

X_shuffled, y_shuffled = shuffle(X, y, random_state=16)

X_train, X_temp, y_train, y_temp = train_test_split(X_shuffled, y_shuffled, test_size=0.3, random_state=16)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=16)


# Data points count in each set
train_count = len(X_train)
val_count = len(X_val)
test_count = len(X_test)

# Labels for the pie chart
labels = ['Training', 'Validation', 'Testing']

# Data points count for each slice
sizes = [train_count, val_count, test_count]

# Colors for each slice
colors = ['lightskyblue', 'lightblue', 'lightgreen']

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'{int(p * sum(sizes) / 100)} ({p:.1f}%)', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.show()
