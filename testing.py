import numpy as np

model = np.load("G:/neural network/loan_approval/loan_model.npz")
weights = model["weights"]
bias = model["bias"]
mean = model["means"]
std = model["stds"]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(X, weights, bias):
    z = np.dot(X,weights) + bias
    y_pred = sigmoid(z)
    return y_pred

def predict_loan(data, weights, bias, means, stds):
    data = np.array(data, dtype=float)
    data = (data - means) / stds
    probability = forward(data, weights, bias)
    return probability


def check_loan():

    income = int(input("Enter annual income (in rupees): "))
    loan_amt = int(input("Enter loan amount (in rupees): "))
    cibil_score = int(input("Enter cibil score: "))
    self_emp = input("Are you self employed ? (y/n): ")
    if (self_emp == "y"):
        self_emp_y = 1
        self_emp_n = 0
    else:
        self_emp_y = 0
        self_emp_n = 1

    data = [income,loan_amt,cibil_score,self_emp_n,self_emp_y]
    loan_approval = predict_loan(data,weights,bias,mean,std)
    print(f"\nLoan approval probability: {loan_approval[0]*100}")

check_loan()