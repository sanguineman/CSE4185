import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    """
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    """
    word0_max = 0 # 확률 변수 X0이 가질 수 있는 최대 값
    word1_max = 0 # 확률 변수 X1이 가질 수 있는 최대 값

    cnt_tuple = [] # 각 텍스트에 대해서 (X0=x0, X1=x1) 튜플 형태로 저장

    for i in range(len(texts)): # iterate each text which is a list of words
      cnt0 = texts[i].count(word0)
      cnt1 = texts[i].count(word1)
      word0_max = max(word0_max, cnt0) # 확률 변수 X0이 가질 수 있는 최대 값, 즉 the maximum number of times that word0 occurs in a given text
      word1_max = max(word1_max, cnt1) # 확률 변수 X1이 가질 수 있는 최대 값, 즉 the maximum number of times that word1 occurs in a given text
      cnt_tuple.append((cnt0,cnt1)) # 각 텍스트에 대해서 (X0=x0, X1=x1) 튜플 형태로 저장


    denom = len(texts) # 분모로 들어갈 모든 경우의 수는 텍스트의 개수와 같다.
    Pjoint = np.zeros((word0_max+1, word1_max+1)) # (word0_max+1, word1_max+1) shape np array

    for i in range(word0_max+1):
      for j in range(word1_max+1):
        Pjoint[i,j] = cnt_tuple.count((i,j)) / denom # calculate the joint distribution
    
    # raise RuntimeError("You need to write this part!")
    return Pjoint


def marginal_distribution_of_word_counts(Pjoint, index):
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other)

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    """
    rows, cols = Pjoint.shape
    if index == 0:
      Pmarginal = np.sum(Pjoint, axis=1).tolist() # calculate the marginal distribution of X0, which means retaining X0 and marginalize X1
    else:
      Pmarginal = np.sum(Pjoint, axis=0).tolist() # calculate the marginal distribution of X1, which means retaining X0 and marginalize X0
    # raise RuntimeError("You need to write this part!")
    return Pmarginal


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs:
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    """
    rows, cols = Pjoint.shape
    Pcond = np.zeros_like(Pjoint) # make the same shape 2d array like Pjoint

    for i in range(rows):
      for j in range(cols):
        if Pmarginal[i] == 0:
          Pcond[i, j] = np.nan # if divide by zero, make it nan
        else:
          Pcond[i,j] = Pjoint[i, j] / Pmarginal[i] # calculate the conditional distribution using the formula

    # raise RuntimeError("You need to write this part!")
    return Pcond


def mean_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    """
    mu = 0.0
    for i in range(len(P)):
      mu += (i * P[i]) # calculate the mean from distribution using the formula

    mu = round(mu, 3) # round to the third 
    # raise RuntimeError("You need to write this part!")
    return mu


def variance_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    """
    mu = mean_from_distribution(P) # calculate the mean first
    var = 0.0
    for i in range(len(P)):
      var += (((i - mu) ** 2) * P[i]) # using the mean, calculate the variance using the formula

    var = round(var, 3) # round to the third
    # raise RuntimeError("You need to write this part!")
    return var


def covariance_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    """
    rows, cols = P.shape
    X0 = marginal_distribution_of_word_counts(P, 0) # get the marginal distribution of X0 of P
    X1 = marginal_distribution_of_word_counts(P, 1) # get the marginal distribution of X1 of P
    mu_X0 = mean_from_distribution(X0) # get the mean of X0
    mu_X1 = mean_from_distribution(X1) # get the mean of X1

    covar = 0.0
    for i in range(rows):
      for j in range(cols):
        covar += ((i - mu_X0) * (j - mu_X1) * P[i, j]) # calculate the covariance using the formula
      
    covar = round(covar, 3) # round to the third

    # raise RuntimeError("You need to write this part!")
    return covar


def expectation_of_a_function(P, f):
    """
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    """
    rows, cols = P.shape
    expected = 0.0
    for i in range(rows):
      for j in range(cols):
        expected += (f(i, j) * P[i, j]) # calculate the expected value of a function using the formula
    
    expected = round(expected, 3) # round to the third
    # raise RuntimeError("You need to write this part!")
    return expected
