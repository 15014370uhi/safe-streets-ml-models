import pandas as pd
import numpy as np
import joblib
import glob
import timeit
import matplotlib.pyplot as plt
from datetime import datetime

# sklearn model
from sklearn.linear_model import SGDClassifier

# sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

# scoring
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

# set array to hold crime sector names
sectors = [
    {  # Northumbria, Durham, Cleveland
        'sector': 'Sector1',
    },
    {  # Cumbria, Lancashire
        'sector': 'Sector2',
    },
    {  # North Yorkshire, West Yorkshire
        'sector': 'Sector3',
    },
    {  # Humberside, South Yorkshire
        'sector': 'Sector4',
    },
    {  # Merseyside, Cheshire
        'sector': 'Sector5',
    },
    {  # Greater Manchester
        'sector': 'Sector6',
    },
    {  # Derbyshire, Nottinghamshire, Lincolnshire, Leicestershire, Northamptonshire
        'sector': 'Sector7',
    },
    {  # West Mercia, Staffordshire, West Midlands, Warwickshire,
        'sector': 'Sector8',
    },
    {  # Gloucestershire, Avon & Somerset, Devon & Cornwall
        'sector': 'Sector9',
    },
    {  # Wiltshire, Dorset, Hampshire (includes isle of wight)
        'sector': 'Sector10',
    },
    {  # Thames Valley, Hertfordshire, Bedforshire
        'sector': 'Sector11',
    },
    {  # Cambridgeshire, Norfolk, Suffolk, Essex
        'sector': 'Sector12',
    },
    {  # Surrey, Sussex, Kent
        'sector': 'Sector13',
    },
    {  # London (including metropolitan)
        'sector': 'Sector14',
    },
]

# function which reads all csv files in a folder and returns a dataframe representation

def getCSVData(aSector):

    # declare variable to hold data frame
    df = pd.DataFrame()

    # declare variable to hold cleaned CSV data
    cleanedData = []

    # define file path
    pathname = "data/" + aSector + "/*.csv"
    allFiles = []

    for file in glob.iglob(pathname, recursive=True):
        allFiles.append(file)

    # for each CSV file in specified path
    for aFile in allFiles:

        # reading CSV data
        CSVData = pd.read_csv(
            aFile, usecols=['Month', 'Latitude', 'Longitude'])

        # filter out any CSV rows with missing data
        CSVData = CSVData.loc[pd.notna(CSVData['Month'])
                              & pd.notna(CSVData['Latitude'])
                              & pd.notna(CSVData['Longitude'])]

        # append data to array of all data
        cleanedData.append(CSVData)

    # convert to data frame
    df = pd.concat(cleanedData)

    # return the data frame
    return df

# function which formats crime data

def formatData(df, clusterModel):

    # get year value from date element
    df['Year'] = df['Month'].apply(lambda month:
                                   datetime.strptime(month, '%Y-%m').year)

    # get month element from date element
    df['Month'] = df['Month'].apply(lambda month:
                                    datetime.strptime(month, '%Y-%m').month)

    # use kmeans to identify cluster for each lat and lon coordinate and assign cluster value
    df['Cluster'] = df.apply(lambda row:
                             clusterModel.predict([[row['Latitude'], row['Longitude']]]).item(0), axis=1)

    # drop lat and lon cols from dataframe
    df = df.drop(['Latitude', 'Longitude'], axis=1)

    # create new dataframe with crime counts per cluster per month per year
    dfCounter = df.value_counts().to_frame('Counts').reset_index()

    # rearrange cols
    dfCounter = dfCounter[['Year', 'Month', 'Cluster', 'Counts']]

    # cast data to int16 to save memory
    dfCounter['Year'] = dfCounter['Year'].astype('int16')
    dfCounter['Month'] = dfCounter['Month'].astype('int16')
    dfCounter['Cluster'] = dfCounter['Cluster'].astype('int16')
    dfCounter['Counts'] = dfCounter['Counts'].astype('int16')
    return dfCounter

# function which returns X and y data sets as numpy arrays


def convertToNP(aDataFrame):

    # convert dataframe to numpy array with floats (dummy)
    npArray = aDataFrame.to_numpy().astype(np.float32)

    # shuffle data
    np.random.shuffle(npArray)

    # return columns as X (Year, Month, Cluster), y (Crime type)
    return npArray[:, :3], npArray[:, 3]

# for each record of crimes in a sector, generate and save logistic regression model
for record in sectors:

    results = {}  # prediction results
    counter = 0  # counter for crime category iteration

    # read crime data for current sector
    print('\n\n=========== ' + record['sector'] + ' ===========')
    print('Reading ' + record['sector'] + ' CSV files...')
    record['df'] = getCSVData(record['sector'])

    # load corresponding KMeans cluster model for current sector
    filename = 'kmini_models/KMini_' + record['sector'] + '.sav'
    record['kMini_Model'] = joblib.load(filename)
    print('loaded', filename + ' KMini model, with clusters:' +
          str(record['kMini_Model'].n_clusters))

    # format data
    print('Formatting data...')
    record['df'] = formatData(record['df'], record['kMini_Model'])

    # get X and y data as X (Year, Month, Cluster), y (Count)
    print('Storing X,y features and values...')
    X, y = convertToNP(record['df'])
    
    # create classifier pipeline and scale X values
    print('Creating Classifier and scaling X values...')    
    classifier = make_pipeline(StandardScaler(), SGDClassifier(random_state=1))

    # scale X feature data
    classifier.scaler = StandardScaler()
    classifier.scaler.fit(X)
    X = classifier.scaler.transform(X)

    # one-hot encode X features
    print('One-Hot encoding X features...')
    clusters = X[:, [2]]
    classifier.encoder = OneHotEncoder(sparse=True)
    one_hot_encoded_location = classifier.encoder.fit(clusters)
    encodedClusters = classifier.encoder.fit_transform(clusters).toarray()

    # reintegrate one-hot encoded values to X
    X = np.hstack((X[:, :2], encodedClusters))

    # split into train and test data for features and target
    print('Test/train data splitting...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # train model
    print('Training model with X_train / y_train...')
    classifier.fit(X_train, y_train)

    # make predictions for the test set
    print('Getting predictions using test data...')
    y_pred_test = classifier.predict(X_test)

    ############## Score #####################        
        
    # Save comparison table of actual crime totals versus predicted to file
    print('\nSaving table of Actual Vs. Predicted...')
    dfResult = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    dfResult.to_csv('comparison_tables/stochasticGD/SGD_comparison_table_' +
                    record['sector'] + '.csv', encoding='utf-8', index=False)
    print('Table saved')

    # save model for this crime sector to file
    filename = 'StochasticGD_Occurrences/StochasticGD_Occurrences_' + record['sector'] + '.sav'
    print('Model Saved as: ' + filename + '\n')
   
    # save graph of prediction versus actual
    print('\nCreating plot of Actual Vs. Predicted values...')   
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred_test, c='blue')
    p1 = max(max(y_pred_test), max(y_test))
    p2 = min(min(y_pred_test), min(y_test))
    plt.title(record['sector'])
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plotFilename = 'comparison_tables/stochasticGD/StochasticGD_Occurrences_' + \
        record['sector'] + '.png'
    plt.savefig(plotFilename)
    # plt.show()
    print('Plot Saved as ' + plotFilename)

    # view R2 score
    print('\nCalculating R2 Score...')    
    r2Score = r2_score(y_test, y_pred_test)
    print('R2 score using test data: ' + str(r2Score)) 
        
    # view Mean Absolute Error score
    print('\nCalculating Mean Absolute Error Score...')    
    meanAbsolError = mean_absolute_error(y_test, y_pred_test) 
    print('Mean Absolute Error score (MAE) using test data: ' + str(meanAbsolError))          
    
    # view Mean Absolute Percentage Error
    print('\nCalculating Mean Absolute Percentage Error...')    
    meanAbsolPercentage = mean_absolute_percentage_error(y_test, y_pred_test) 
    print('Mean Absolute Percentage Error (MAPE) using test data: ' + str(meanAbsolPercentage)) 
              
    # Explained Variance Score
    print('\nCalculating Explained Variance...')   
    explainedVariance = explained_variance_score(y_test, y_pred_test) 
    print('Explained Variance score using test data: ' + str(explainedVariance))     
     
    ######################################################

print('\nOperation complete')
print('\n========--- FINISHED ---========')
