import pandas as pd
import numpy as np
import joblib
import glob
import timeit
from datetime import datetime

# sklearn model
from sklearn.linear_model import LogisticRegression

# sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


# set array to hold crime sector names
sectors = [  
    { # Northumbria, Durham, Cleveland
        'sector': 'Sector1', 
    },
    { # Cumbria, Lancashire  
        'sector': 'Sector2', 
    },
    { # North Yorkshire, West Yorkshire 
        'sector': 'Sector3', 
    },
    { # Humberside, South Yorkshire
        'sector': 'Sector4', 
    },
    { # Merseyside, Cheshire   
        'sector': 'Sector5', 
    },
    { # Greater Manchester
        'sector': 'Sector6', 
    },  
    { # Derbyshire, Nottinghamshire, Lincolnshire, Leicestershire, Northamptonshire
        'sector': 'Sector7', 
    },    
    { # West Mercia, Staffordshire, West Midlands, Warwickshire, 
        'sector': 'Sector8', 
    },    
    { # Gloucestershire, Avon & Somerset, Devon & Cornwall
        'sector': 'Sector9', 
    },    
    { # Wiltshire, Dorset, Hampshire (includes isle of wight)
        'sector': 'Sector10', 
    },    
    { # Thames Valley, Hertfordshire, Bedforshire
        'sector': 'Sector11', 
    },    
    { # Cambridgeshire, Norfolk, Suffolk, Essex
        'sector': 'Sector12', 
    },    
    { # Surrey, Sussex, Kent
        'sector': 'Sector13', 
    },    
    { # London (including metropolitan)
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
            aFile, usecols=['Month', 'Crime type', 'Latitude', 'Longitude'])

        # filter out any CSV rows with missing data
        CSVData = CSVData.loc[pd.notna(CSVData['Month'])
                              & pd.notna(CSVData['Crime type'])
                              & pd.notna(CSVData['Latitude'])
                              & pd.notna(CSVData['Longitude'])]

        # append data to array of all data
        cleanedData.append(CSVData)

    # convert to data frame
    df = pd.concat(cleanedData)

    # return the data frame
    return df

# function which converts a crime category to a number
def getCrimeValue(aCrime):
    if(aCrime == 'Anti-social behaviour'):
        return 0  # anti-social behaviour   
    if(aCrime == 'Burglary'):
        return 1  # burglary
    if(aCrime == 'Criminal damage and arson'):
        return 2  # criminal damage
    if(aCrime == 'Drugs'):
        return 3  # drugs
    if(aCrime == 'Possession of weapons'):
        return 4  # weapons    
    if(aCrime == 'Public order'
       or aCrime == 'Other crime'):
        return 5  # public order    
    if(aCrime == 'Robbery'):
        return 6 # robbery
    if(aCrime == 'Shoplifting'):
        return 7 # shop lifting 
    if(aCrime == 'Theft from the person'
       or aCrime == 'Bicycle theft'
       or aCrime == 'Other theft'):
        return 8  # theft
    if(aCrime == 'Vehicle crime'):
        return 9  # vehicle
    if(aCrime == 'Violent crime'
       or aCrime == 'Violence and sexual offences'):
        return 10  # violent crime

# function which returns the crime category for a given crime value
def getCrimeCategory(aCrimeValue):
    if(aCrimeValue == 0):
        return 'Anti-social behaviour'  # anti-social behaviour   
    if(aCrimeValue == 1):
        return 'Burglary'  # burglary
    if(aCrimeValue == 2):
        return 'Criminal damage and arson'  # criminal damage
    if(aCrimeValue == 3):
        return 'Drugs'  # drugs
    if(aCrimeValue == 4):
        return 'Possession of weapons'  # weapons       
    if(aCrimeValue == 5):
        return 'Public order'  # public order       
    if(aCrimeValue == 6):
        return 'Robbery' # robbery
    if(aCrimeValue == 7):
        return 'Shoplifting' # shop lifting 
    if(aCrimeValue == 8):
            return 'Theft'  # theft
    if(aCrimeValue == 9):
        return 'Vehicle crime'  # vehicle
    if(aCrimeValue == 10):    
        return 'Violent crime'  # violent crime
             

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

    # convert crime categories into numerical values
    df['Crime type'] = df['Crime type'].apply(getCrimeValue)    

    # rearrange cols
    df = df[['Year', 'Month', 'Cluster', 'Crime type']]    

    # cast data to int16 to save memory
    df['Year'] = df['Year'].astype('int16')
    df['Month'] = df['Month'].astype('int16')
    df['Cluster'] = df['Cluster'].astype('int16')
    df['Crime type'] = df['Crime type'].astype('int16')  

    return df

# function which returns X and y data sets as numpy arrays
def convertToNP(aDataFrame):

    # convert dataframe to numpy array with floats (dummy)
    npArray = aDataFrame.to_numpy().astype(np.float32)

    # shuffle data
    np.random.shuffle(npArray)

    # return columns as X (Year, Month, Cluster), y (Crime type)
    return npArray[:, :3], npArray[:, 3]

# start timer
start = timeit.default_timer() 

# for each record of crimes in a sector, generate and save logistic regression model
for record in sectors:
    
    results = {} # prediction results
    counter = 0 # counter for crime category iteration
    mostActiveCrime = '' # store name of most prevalent crime to occur
    
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
    
     # define classifier pipeline 
    print('Creating classifier and scaling X values...')   
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        solver='lbfgs', max_iter=1000, random_state=0))   

    # get X and y data as X (Year, Month, Cluster), y (Crime type)
    print('Storing X,y features and values...')   
    X, y = convertToNP(record['df'])

    # # define classifier pipeline 
    # print('Creating classifier and scaling X values...')   
    # classifier = make_pipeline(StandardScaler(), LogisticRegression(
    #     solver='lbfgs', max_iter=1000, random_state=0))   
    
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
    print('Getting prediction percentage score using test data...')   
    y_pred_test = classifier.predict(X_test)

    # view accuracy score
    accuracyPercentage = accuracy_score(y_test, y_pred_test)
    print('Accuracy score using test data: ' + str(accuracyPercentage))   

    # testing prediction probability with test data
    print('Predicting crime distribution ratio using test data...')   
    predictions = classifier.predict_proba(X_test)[0] 
    highestPercentage = f'{(np.amax(predictions) * 100):.2f}'

    # display prediction   
    for percentage in predictions:
        aPercentage = f'{(percentage * 100):.2f}'

        # get name of highest percentage crime category
        if aPercentage == highestPercentage:
            mostActiveCrime = getCrimeCategory(counter)

        aPercentage = str(aPercentage + '%')
        crimeCategory = getCrimeCategory(counter)
        results[crimeCategory] = aPercentage
        counter += 1

    print('\nMost active crime category: ' + mostActiveCrime +
          ' ' + str(highestPercentage) + '%')
    print(str(results))

    # save model for this crime sector to file
    filename = 'logisticRegression_Ratio_Models/LR_Ratio_' + record['sector'] + '.sav'
    joblib.dump(classifier, filename)
    print('Saved logistic regression model: ' + filename)
    
    # cross validation score
    # print('Calculating cross validation score...')
    # start = timeit.default_timer()
    # cv_results = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    # print(cv_results.mean())
    # stop = timeit.default_timer()
    # timeTaken = f'{(stop - start):.2f}' #stop - start
    # print('Complete in: ' + str(timeTaken) + ' seconds\n') #{timeTaken:.2f}    
    # test load of the model from the file
    #myClassifier = joblib.load(filename)
    # print(myClassifier)

# stop timer
stop = timeit.default_timer()
timeTaken = f'{(stop - start):.2f}'
print('Operation complete in: ' + str(timeTaken) + ' seconds\n')  
print('========--- FINISHED ---========')
