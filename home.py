import flask
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.sparse import csr_matrix
from IPython.display import display_html
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import implicit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


app =flask.Flask(__name__, template_folder='templates')

#change path to yours
path = '/Users/DELL/Desktop/project-1/'
dataset = 'Data.xlsx'


def read_data():
    data = pd.read_excel(path+dataset)
    return data


data = read_data()

all_products= [data['Product'] [i] for i in range (len(data['Product']))]

data['CustID'] = data['CustID'].astype(str)
data['gender'] = data['gender'].replace({'male': 'Male', 'female': 'Female'})

item_lookup = data[['Product', 'ProductType']] # Only get unique item/description pairs
#item_lookup['Product'] = item_lookup.Product.astype(str) # Encode as strings for future lookup ease

data =data[['Product','ProductType','age', 'Amount', 'CustID']] # Get rid of unnecessary info
grouped_cleaned = data.groupby(['CustID', 'Product']).sum().reset_index() # Group together
grouped_purchased = grouped_cleaned.query('Amount > 0')

grouped_purchased.CustID=grouped_purchased.CustID.astype(str)
customers = list(np.sort(grouped_purchased.CustID.unique())) # Get our unique customers
products = list(grouped_purchased.Product.unique()) # Get our unique products that were purchased
amount = list(grouped_purchased.Amount) # All of our purchases #replace amount binned

rows = grouped_purchased.CustID.astype('category').cat.codes 
# Get the associated row indices
cols = grouped_purchased.Product.astype('category').cat.codes 
# Get the associated column indices
purchases_sparse = sparse.csr_matrix((amount, (rows, cols)), shape=(len(customers), len(products)))

matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))

def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered


product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)

def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):
    conf = (alpha*training_set) # To allow the matrix to stay sparse, I will add one later when each row is taken 
                                # and converted to dense. 
    num_user = conf.shape[0]
    num_item = conf.shape[1] # Get the size of our original ratings matrix, m x n
    
    # initialize our X/Y feature vectors randomly with a set seed
    rstate = np.random.RandomState(seed)
    
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) # Random numbers in a m x rank shape
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size))) # Normally this would be rank x n but we can 
                                                                 # transpose at the end. Makes calculation more simple.
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    lambda_eye = lambda_val * sparse.eye(rank_size) # Our regularization term lambda*I. 
    
    # We can compute this before iteration starts. 
    
    # Begin iterations
    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa
        # Compute yTy and xTx at beginning of each iteration to save computing time
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        # Being iteration to solve for X based on fixed Y
        for u in range(num_user):
            conf_samp = conf[u,:].toarray() # Grab user row from confidence matrix and convert to dense
            pref = conf_samp.copy() 
            pref[pref != 0] = 1 # Create binarized preference vector 
            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term, which is just CuI since we never added 1
            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term, where we add the eye back in
                                                      # Cu - I + I = Cu
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper  
        # Begin iteration to solve for Y based on fixed X 
        for i in range(num_item):
            conf_samp = conf[:,i].T.toarray() # transpose to get it in row format and convert to dense
            pref = conf_samp.copy()
            pref[pref != 0] = 1 # Create binarized preference vector
            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term, which is just CiI since we never added 1
            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, equation 5 from the paper
    # End iterations
    return X, Y.T 

user_vecs, item_vecs = implicit_weighted_ALS(product_train, lambda_val = 0.1, alpha = 15, iterations = 1,
                                            rank_size = 20)
user_vecs[0,:].dot(item_vecs).toarray()[0,:5]
alpha = 15
user_vecs, item_vecs = implicit.alternating_least_squares((product_train*alpha).astype('double'), 
                                                          factors=20, 
                                                          regularization = 0.1, 
                                                         iterations = 50)


def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))

calc_mean_auc(product_train, product_users_altered, 
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], product_test)

customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix
products_arr = np.array(products) # Array of product IDs from the ratings matrix


def rec_items(customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup,num_items = 10):

        cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id
        pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
        pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
        pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
        rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot product of user vector and all item vectors
        # Scale this recommendation vector between 0 and 1
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
        recommend_vector = pref_vec*rec_vector_scaled 
        # Items already purchased have their recommendation multiplied by zero

        product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
        # of best recommendations
        rec_list = [] # start empty list to store items
        for index in product_idx:
            code = item_list[index]
            rec_list.append([code, item_lookup.ProductType.loc[item_lookup.Product == code].iloc[0]]) 
        # Append our descriptions to the list
        codes = [item[0] for item in rec_list]
        descriptions = [item[1] for item in rec_list]
        final_frame = pd.DataFrame({'Product': codes, 'ProductType': descriptions}) # Create a dataframe 
        return final_frame[['Product', 'ProductType']]


@app.route('/', methods=['GET','POST'])
def home():
    if flask.request.method == 'GET':
        return (flask.render_template('home.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['CustID']

       
        if m_name not in customers_arr:
            return (flask.render_template('home.html', name =m_name))
        else: 
            num_items = 10
            result_final = rec_items(m_name, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup,num_items)
            names =[]
            producttype =[]
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                producttype.append(result_final.iloc[i][1])

            
            return flask.render_template('predict-page.html',  product_name =names, productT=producttype, search_name =m_name)



        

if __name__ == "__main__":

    app.run(debug=True)
