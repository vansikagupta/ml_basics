# ml_basics


# Data preprocessing:
	This is an essential step which makes the data ready to be fed into ML algorithms.
  * Importing the libraries
  * Importing the data
  * Handling missing data: either remove the rows or replace with some aggregate column values(mean, mode, median, etc.) 
  * sklearn is the ML library of python, you get all imp methods for all neccessary steps
  * Handling categorical data: encode text data to numbers since ML models work with mathematical equations
  * This can be done in many ways depending on the dataset and ML model.   Replacing text data with numbers, Binary Encoding, 
    Using LabelEncoder, Binary Encoding and many more.
  * OneHotEncoder adds new columns of the same number as the number of categories
  * Splitting the data set into train and test data since you need to test how accurately the ML model has learned 
  * Feature scaling: it is required if working with some ML models and is important because otherwise some features with higher 
    range of values might dominate the results. For an example while computing the Euclidean distance 
