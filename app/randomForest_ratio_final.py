import pandas as pd
import numpy as np 

import joblib
from datetime import datetime
import glob
import timeit
import matplotlib.pyplot as plt

#models
from sklearn.ensemble import RandomForestClassifier

#utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

#scoring
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


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


#function which reads all csv files in a folder and returns a dataframe representation
def getCSVData(aSector):
    
    #declare variable to hold data frame
    df = pd.DataFrame()
    
    #declare variable to hold cleaned CSV data
    cleanedData = []      
             
    pathname = "data/" + aSector + "/*.csv"      
       
    allFiles = []    
    
    for file in glob.iglob(pathname, recursive=True):
        allFiles.append(file)    
    
    #for each CSV file in specified path
    for aFile in allFiles:   
         
        #reading CSV data
        CSVData = pd.read_csv(aFile, usecols=['Month', 'Crime type', 'Latitude', 'Longitude'])     
        
        #filter out any CSV rows with missing data          
        CSVData = CSVData.loc[pd.notna(CSVData['Month']) 
                        & pd.notna(CSVData['Crime type'])
                        & pd.notna(CSVData['Latitude'])
                        & pd.notna(CSVData['Longitude'])]
        
        #append data to array of all data
        cleanedData.append(CSVData)
        
    #convert to data frame
    df = pd.concat(cleanedData)     
    
    #return the data frame
    return df

#function which converts a crime category to a number value
def getCrimeValue(aCrime):     
 if(aCrime == 'Anti-social behaviour'):    
     return 0 #anti-social behaviour
 if(aCrime == 'Bicycle theft'
    or aCrime == 'Other theft'
    or aCrime == 'Shoplifting'):      
     return 1 #theft
 if(aCrime == 'Burglary'):   
     return 2 #burglary
 if(aCrime == 'Criminal damage and arson'):   
     return 3 #criminal damage
 if(aCrime == 'Drugs'):   
     return 4 #drugs
 if(aCrime == 'Public order' 
    or aCrime == 'Other crime'):   
     return 5 #public order
 if(aCrime == 'Possession of weapons'):   
     return 6 #weapons
 if(aCrime == 'Violent crime' 
    or aCrime == 'Theft from the person'
    or aCrime == 'Robbery'   
    or aCrime == 'Violence and sexual offences'): 
     return 7 #violent crime   
 if(aCrime == 'Vehicle crime'):   
     return 8 #vehicle  

#returns the crime category for a given crime value
def getCrimeCategory(aCrimeValue):     
    if(aCrimeValue == 0):    
        return 'Anti-social behaviour' #anti-social behaviour
    if(aCrimeValue == 1):      
        return 'Theft' #theft
    if(aCrimeValue == 2):   
        return 'Burglary' #burglary
    if(aCrimeValue == 3):   
        return 'Criminal damage and arson' #criminal damage
    if(aCrimeValue == 4):   
        return 'Drugs' #drugs
    if(aCrimeValue == 5):   
        return 'Public order' #public order
    if(aCrimeValue == 6):   
        return 'Possession of weapons' #weapons
    if(aCrimeValue == 7): 
        return 'Violent crime'  #violent crime   
    if(aCrimeValue == 8):   
        return 'Vehicle crime' #vehicle  

#format data
def formatData(df, clusterModel):
    
    #get year value from date element
    df['Year'] = df['Month'].apply(lambda month: 
        datetime.strptime(month, '%Y-%m').year)

    #get month element from date element
    df['Month'] = df['Month'].apply(lambda month: 
        datetime.strptime(month, '%Y-%m').month)

    # use kmeans to identify cluster for each lat and lon coordinate and assign cluster value
    df['Cluster'] = df.apply(lambda row: 
        clusterModel.predict([[row['Latitude'], row['Longitude']]]).item(0), axis=1)

    #drop lat and lon cols from dataframe
    df = df.drop(['Latitude', 'Longitude'], axis=1)

    #convert crime categories into numerical values
    df['Crime type'] = df['Crime type'].apply(getCrimeValue)

    #rearrange cols
    df = df[['Year', 'Month', 'Cluster', 'Crime type']]

    df['Year'] = df['Year'].astype('int16')
    df['Month'] = df['Month'].astype('int16')
    df['Cluster'] = df['Cluster'].astype('int16')
    df['Crime type'] = df['Crime type'].astype('int16')

    return df
    #752 seconds


#get X and y data sets
def convertToNP(aDataFrame):    
    
    #convert dataframe to numpy array with floats (dummy)
    npArray = aDataFrame.to_numpy().astype(np.float32) 
    
    #shuffle data
    np.random.shuffle(npArray)  
    
    #return columns as X (Year, Month, Cluster), y (Crime type)   
    return npArray[:, :3], npArray[:, 3]  

#store CSV file data as dataframe for each sector
for record in sectors:
    print('\n\n=========== ' + record['sector'] +' ===========')
    print('Reading ' + record['sector'] + ' CSV files...') 
    start = timeit.default_timer()
    record['df'] = getCSVData(record['sector'])    
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}' #stop - start     
    print('Complete in: ' + str(timeTaken) + ' seconds\n') #{timeTaken:.2f}
   
    # load KMeans cluster model for each sector
    filename = 'kmini_models/KMini_' + record['sector'] + '.sav'
    record['kMini_Model'] = joblib.load(filename)
    print('loaded', filename + ' KMini model, with clusters:' +
          str(record['kMini_Model'].n_clusters))
    
    #format data
    print('Formatting data...')
    start = timeit.default_timer()
    record['df'] = formatData(record['df'], record['kMini_Model'])  
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start 
    print('Complete in: ' + str(timeTaken) + ' seconds\n')
    
    #get X and y data
    print('Storing X,y features and values...')
    start = timeit.default_timer()
    X,y = convertToNP(record['df'])
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start  
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    #create classifier pipeline and scale X values 
    print('Creating classifier and scaling X values...')  
    start = timeit.default_timer()
    classifier = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=5, random_state=0)) 
    classifier.scaler = StandardScaler()
    classifier.scaler.fit(X)
    X = classifier.scaler.transform(X)
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start    
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    #one-hot encode
    print('One-Hot encoding X features...')
    start = timeit.default_timer()
    clusters = X[:, [2]]
    classifier.encoder = OneHotEncoder(sparse=True)
    one_hot_encoded_location = classifier.encoder.fit(clusters)
    encodedClusters = classifier.encoder.fit_transform(clusters).toarray()    
    X = np.hstack((X[:, :2], encodedClusters)) #reintegrate one-hot encoded values to X
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    #split into train and test data for features and target
    print('Test/train data splitting...')
    start = timeit.default_timer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start
    print('Complete in: ' + str(timeTaken) + ' seconds\n')  
    
    #train model
    print('Training model with X_train / y_train...')
    start = timeit.default_timer()
    classifier.fit(X_train, y_train)
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start  
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    # Make predictions for the test set
    print('Getting Y prediction using X_test data...')
    start = timeit.default_timer()
    y_pred_test = classifier.predict(X_test)
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start 
    print('Complete in: ' + str(timeTaken) + ' seconds\n')            
        
    # view R2 score
    print('Calculating R2 Score...')
    start = timeit.default_timer()
    r2Score = r2_score(y_test, y_pred_test) 
    print('R2 score using test data: ' + str(r2Score))  #0.0 is best 
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start 
    print('Complete in: ' + str(timeTaken) + ' seconds\n')
    
    # view Mean Absolute Error score
    print('Calculating Mean Absolute Error Score...')
    start = timeit.default_timer()
    meanAbsolError = mean_absolute_error(y_test, y_pred_test) 
    print('Mean Absolute Error score using test data: ' + str(meanAbsolError))  #0.0 is best 
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start 
    print('Complete in: ' + str(timeTaken) + ' seconds\n')
    
    # View accuracy score
    print('Calculating Accuracy Score...')
    start = timeit.default_timer()
    accuracyScore = accuracy_score(y_test, y_pred_test) 
    print('Accuracy Score using test data: ' + str(accuracyScore))
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start 
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    # Cross-Validation Score
    print('Calculating Cross Validation score...')
    start = timeit.default_timer()
    crossValidationScore = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    print(crossValidationScore.mean())
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start   
    print('Complete in: ' + str(timeTaken) + ' seconds\n')   

    # Mean Score
    print('Calculating Mean Score...')
    start = timeit.default_timer()
    meanScore = classifier.score(X_test, y_test)
    print(meanScore)
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start   
    print('Complete in: ' + str(timeTaken) + ' seconds\n')   
    
    #testing the prediciton of probability
    print('Predict probabilities for crime types using X_test data')
    start = timeit.default_timer()
    predictions = classifier.predict_proba(X_test)[0] 
    stop = timeit.default_timer()  
    timeTaken = f'{(stop - start):.2f}'  # stop - start   
    print('Complete in: ' + str(timeTaken) + ' seconds\n')

    results = {} 
    counter = 0
    mostLikelyCrime = ''

    highestPercentage = f'{(np.amax(predictions) * 100):.2f}'  
    
    print('Crime prediction percentages for this month: ' + '\n') #TODO change to daily/weekly
    for percentage in predictions:  
        aPercentage = f'{(percentage * 100):.2f}' 
        
        #get name of highest percentage crime category       
        if aPercentage == highestPercentage: 
            mostLikelyCrime = getCrimeCategory(counter)
            
        aPercentage = str(aPercentage + '%')
        crimeCategory = getCrimeCategory(counter)
        results[crimeCategory] = aPercentage
        counter += 1   

    print('\nMost likely crime: ' + mostLikelyCrime + ' ' + str(highestPercentage) + '%')
    print(str(results))
    
    #save model to file    
    filename = 'RandomForest_Ratios/RandomForest_' + record['sector'] + '.sav'  
    joblib.dump(classifier, filename)
    print('Saved random forest: ' + filename)      

print('========--- FINISHED ---========')