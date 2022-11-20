# examples of unit test cases
from sklearn import model_selection
from sklearn.model_selection import train_test_split

def sum(x, y):
    return x+y

def test_sum():
    x = 5
    y = 7
    z = sum(x,y)
    expected_z = 12
    assert z == expected_z

def test_equal():
    assert 1==1
    
def test_len():
    z1=test_random_state()
    expected_z1=test_random_state2()
    assert z1==expected_z1

def test_random_state():
    
    X_data = range(10)
    y_data = range(10)

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3) 
    return len(X_train)

def test_random_state2():
    
    X_data = range(10)
    y_data = range(10)

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state=0) # zero or any other integer
    return len(X_train)   
    




