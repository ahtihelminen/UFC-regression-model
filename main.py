import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import random


class UfcBettingPredictor:
  def __init__(self):
    self.dataframe = pd.read_csv('./data/ufcData.csv')
    self.errorsDf = pd.DataFrame()
    self.models = {}
    self.moneyWon = 0
    self.moneyBetted = 0

  def printDf(self):
    print('-----------------------------------'*4)
    print(self.dataframe.head())
    print(self.dataframe.shape)
    print('-----------------------------------'*4)

  def printHead(self, df):
    print('-----------------------------------'*3)
    print(df.head())
    print('-----------------------------------'*3)

  def filterDf(self, columns):
    self.dataframe = self.dataframe[columns]

  def fillNa(self, value):
    self.value = value
    self.dataframe = self.dataframe.fillna(self.value)

  def createDiffColumns(self):
    for index, row in self.dataframe.iterrows():
      
      # Calculation are written so that favorable outcome for red fighter
      # is positive and for blue fighter is negative

      # Assuming that having more rounds(=experience) is better
      self.dataframe.loc[
        index,
        'dTotal_rounds_fought'
      ] = row['R_total_rounds_fought'] - row['B_total_rounds_fought'] 
      
      # Assuming that having more wins is better
      self.dataframe.loc[
        index,
        'dWins'
      ] = row['R_wins'] - row['B_wins']

      # Assuming that having more losses is worse
      self.dataframe.loc[
        index,
        'dLosses'
      ] = row['B_losses'] - row['R_losses']
      
      # If both have win streak we calculate the difference
      # between win streaks
      R_win_streak = self.dataframe['R_current_win_streak']
      B_win_streak = self.dataframe['B_current_win_streak']

      if B_win_streak[index] and R_win_streak[index]:
        self.dataframe.loc[
          index,
          'dWin_streak'
        ] = row['R_current_win_streak'] - row['B_current_win_streak']
      
      # If red corner doesn't have a win streak we'll count it's 
      # lose streak in favor of the blue corner
      elif B_win_streak[index] and not R_win_streak[index]:
        self.dataframe.loc[
          index,
          'dWin_streak'
        ] = -row['R_current_lose_streak'] - row['B_current_win_streak']

      # If blue corner doesn't have a win streak we'll count 
      # it's lose streak
      # in favor of the red corner
      elif not B_win_streak[index] and R_win_streak[index]:
        self.dataframe.loc[
          index,
          'dWin_streak'
        ] = row['R_current_win_streak']+row['B_current_lose_streak']
      
      # If both have lose streaks we'll subtract the red corner's lose streak
      # from the blue corner's lose streak
      # -> we end up >0 difference if red corner has shorter lose streak
      # and <0 if blue corner has shorter lose streak
      else:
        self.dataframe.loc[
          index,
          'dWin_streak'
        ] = row['B_current_lose_streak'] - row['R_current_lose_streak']

      # If both have ranking we calculate the difference between
      # rankings (0 is the best rank, 15 worst ->  diff maps from -15 to 15)
      R_weightclass_rank = self.dataframe['R_match_weightclass_rank']
      B_weightclass_rank = self.dataframe['B_match_weightclass_rank']

      if B_weightclass_rank[index] and R_weightclass_rank[index]:
        self.dataframe.loc[
          index,
          'dRanking'
        ] = row['B_match_weightclass_rank'] - row['R_match_weightclass_rank']

      # For non ranked fighters we'll assume the difference to 0 
      else:
        self.dataframe.loc[index, 'dRanking'] = 0

  def moneylineToDecimal(self, column):
    self.dataframe[column] = self.dataframe[column].apply(
      lambda x: x/100 + 1 if x > 0 else 100/abs(x) + 1
    )

  def calculateExpectedValue(self, predictedOdds, givenOdds, betSize):
    return predictedOdds/givenOdds*betSize-betSize - (1-predictedOdds)*betSize
  
  def calculateResult(self, givenOdds, rowNumber, betSize):
    if self.dataframe['Winner'][rowNumber] == 'Red':
      self.moneyWon += betSize * (givenOdds - 1)
    else:
      self.moneyWon -= betSize


  def splitSets(self, features, labels, setSize, random_state=16):
    # First we shuffle the sets
    features_shuffled, labels_shuffled = shuffle(
      features,
      labels,
      random_state=random_state
    )

    # 1-setSize of the data is used for training, setSize for temporary sets
    features_train,features_temp, labels_train, labels_temp = train_test_split(
      features_shuffled,
      labels_shuffled,
      test_size=setSize,
      random_state=random_state
    )

    # Temporary sets are split in half for validation and test sets
    # both 15 % of initial data
    features_test, features_val, labels_test, labels_val = train_test_split(
      features_temp,
      labels_temp,
      test_size=0.5,
      random_state=random_state
    )

    self.printHead(features_train)
    self.printHead(labels_train)
    self.printHead(features_val)
    self.printHead(labels_val)
    self.printHead(features_test)
    self.printHead(labels_test)


    return (features_train, labels_train),\
           (features_val, labels_val),\
           (features_test, labels_test)

  def drawPieChart(self, labels, sizes, title):
    # Colors for each slice
    colors = ['lightskyblue', 'lightblue', 'lightgreen']

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
      sizes,
      labels=labels,
      colors=colors,
      autopct=lambda p: f'{int(p * sum(sizes) / 100)} ({p:.1f}%)', startangle=140
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    # Display the pie chart
    plt.show()

  def drawParallelCoordinates(self, X):
    ss = StandardScaler()
    
    scaledDf = ss.fit_transform(X)
    scaledDf = pd.DataFrame(scaledDf, columns=X.columns)


    scaledDf['categorized_odds'] = pd.cut(
      self.dataframe['R_odds'],
      bins=[1.00, 2.00, 2.01, 100],
      labels=['red', 'even', 'blue'],
      right=False  
    )

    self.printHead(scaledDf)

    plot = pd.plotting.parallel_coordinates(
      scaledDf,
      'categorized_odds',
      color=('r', 'b' , 'c', 'g')
    )

    #hide 'nan' from legend
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(handles[:2], labels[:2])
    plt.show()

  def drawPairWiseHeatmap(self):
    
    numberData = self.dataframe[[
      'R_odds',
      'dTotal_rounds_fought',
      'dWins',
      'dLosses',
      'dWin_streak',
      'dRanking'
    ]]
    numberDataCorr = numberData.corr()
      
    plt.figure(figsize=(8, 6))
    # Hides the upper triangle of the heatmap 
    # lower triangle is enough since matrix is symmetric
    hideUpperTriangle = np.triu(numberDataCorr) 
    sns.heatmap(
      numberDataCorr,
      annot=True,
      cmap="coolwarm",
      fmt=".2f",
      linewidths=0.5,
      annot_kws={"size": 10},
      mask=hideUpperTriangle
    )

    # Add labels and a title
    plt.title("Correlation Heatmap")
    plt.xlabel("Variables")
    plt.ylabel("Variables")

    # Show the heatmap
    plt.show()
    
  def polynomialModel(self, X_train, y_train, X_val, y_val, degreeList):
    
    for degree in degreeList:
      poly = PolynomialFeatures(degree=degree)
      X_train_poly = poly.fit_transform(X_train)

      model = LinearRegression(fit_intercept=False)
      model.fit(X_train_poly, y_train)

      y_train_pred = model.predict(X_train_poly)
      tr_error = mean_squared_error(y_train, y_train_pred)
      self.errorsDf.loc[degree, 'Training error'] = round(tr_error, 5)
      
      X_val_poly = poly.transform(X_val)
      y_val_pred = model.predict(X_val_poly)
      val_error = mean_squared_error(y_val, y_val_pred)
      self.errorsDf.loc[degree, 'Validation error'] = round(val_error, 5)

      self.models[degree] = model

    # Checking degree corresponding the smallest validation error
    self.minErrorDegree = self.errorsDf['Validation error'].idxmin()

  def linearModel(self, X_train, y_train, X_val, y_val):
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    tr_error = mean_squared_error(y_train, y_train_pred)
    self.errorsDf.loc['Linear', 'Training error'] = round(tr_error, 5)

    y_val_pred = model.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_pred)
    self.errorsDf.loc['Linear', 'Validation error'] = round(val_error, 5)
    
    self.models['Linear'] = model

    
  def testBetting(self, X_test, y_test):
    
    self.moneyWon = 0    
    betSize = 10
    results = pd.DataFrame()
 
    for row in self.dataframe.iterrows():
      self.calculateResult(row[1]['R_odds'], row[0], betSize)
    results.loc[
      'Always red',
      'W%'
    ] = 100*round(self.moneyWon/(len(y_test)*10), 3)

    self.moneyWon = 0
    self.randomList = []
    for i in range(10):
      for row in self.dataframe.iterrows():
        if self.calculateExpectedValue(
          random.randrange(1, 10),
          row[1]['R_odds'], betSize
        ) > 0:
          self.moneyBetted += betSize
          self.calculateResult(row[1]['R_odds'], row[0], betSize)
      self.randomList.append(100*round(self.moneyWon/self.moneyBetted, 3))
    results.loc[
      'Random',
      'W%'
    ] = round(sum(self.randomList)/len(self.randomList), 3)
    
    self.moneyWon = 0
    self.moneyBetted = 0

    model = self.models[self.minErrorDegree]

    poly = PolynomialFeatures(degree=self.minErrorDegree)
    X_test_poly = poly.fit_transform(X_test)
    y_test_pred = model.predict(X_test_poly)

    testError = mean_squared_error(y_test, y_test_pred)
    results.loc[self.minErrorDegree, 'Test error'] = testError

    for predictedOdds,\
        givenOdds,\
        rowNumber in zip(y_test_pred, y_test, y_test.index):
      if self.calculateExpectedValue(predictedOdds, givenOdds, betSize) > 0:
        self.moneyBetted += betSize
        self.calculateResult(givenOdds, rowNumber, betSize)
    results.loc[
      self.minErrorDegree,
      'W%'
    ] = 100*round(self.moneyWon/self.moneyBetted, 3)
    results.fillna('', inplace=True)
    print('-----------------------------------'*4)
    print(results)



  def main(self):
    self.printDf()
    
    self.filterDf([
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
    ])
    self.printDf()
    
    self.fillNa(-1)

    self.moneylineToDecimal('R_odds')
    self.moneylineToDecimal('B_odds')
    self.createDiffColumns()
    self.printDf()

    self.filterDf([
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
    ])
    self.printDf()


    features = self.dataframe[[
      'dTotal_rounds_fought',
      'dWins',
      'dLosses',
      'dWin_streak',
      'dRanking'
    ]]
    labels = self.dataframe['R_odds']

    datasetArray = self.splitSets(features, labels, 0.3, 16)

    self.drawPieChart(
      ['Training', 'Validation', 'Testing'],
      [len(datasetArray[0][0]),
       len(datasetArray[1][0]),
       len(datasetArray[2][0])],
       'Data points count in each set'
    )

    degrees = [2, 3, 4, 5, 6]

    self.drawParallelCoordinates(features)
    self.drawPairWiseHeatmap()

    self.linearModel(
      datasetArray[0][0],
      datasetArray[0][1],
      datasetArray[1][0],
      datasetArray[1][1]
    )

    self.polynomialModel(
      datasetArray[0][0],
      datasetArray[0][1],
      datasetArray[1][0],
      datasetArray[1][1],
      degrees
    )

    print(self.errorsDf)

    self.testBetting(
      datasetArray[2][0],
      datasetArray[2][1]
    )

predictor = UfcBettingPredictor()
predictor.main()
