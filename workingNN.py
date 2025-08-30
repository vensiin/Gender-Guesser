import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derive_sigmoid(x):
    fx = sigmoid(x)
    derived = fx * (1 - fx)

    return derived

def MSE_LOSS(y_true, y_pred):
    loss = ((y_true - y_pred) ** 2).mean()
    return loss

def converyHW_array(user_height, user_weight):
    #Usually we subtract by the mean but we are just following the guide right now, and he subbed by 135 and 66)
    new_height = user_height - 66
    new_weight = user_weight - 135

    user_measurements = np.array([new_weight, new_height])

    return user_measurements



class OurNN:
    # Class variables
    def __init__(self):
        # Assign initial random weights
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()

        # Assign initial random bias
        self.b1 = np.random.randn()
        self.b2 = np.random.randn()
        self.b3 = np.random.randn()

    # Retrieve Weights
    def get_weights_and_bias(self):

        print(f"Initial W1: {self.w1}")
        print(f"Initial W2: {self.w2}")
        print(f"Initial W3: {self.w3}")
        print(f"Initial W4: {self.w4}")
        print(f"Initial W5: {self.w5}")
        print(f"Initial W6: {self.w6}")
        print(f"Initial B1: {self.b1}")
        print(f"Initial B2: {self.b2}")
        print(f"Initial B3: {self.b3}")

    # Returns prediction
    def feed_forward(self, inputs):
        h1 = sigmoid(self.w1 * inputs[0] + self.w2 * inputs[1] + self.b1)
        h2 = sigmoid(self.w3 * inputs[0] + self.w4 * inputs[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    # How we train our model
    def train(self, data, y_true_values):

        epochs = 0
        learn_rate = 0.1
        stop_loss = 0.01
        y_preds = np.apply_along_axis(self.feed_forward, 1, data)
        loss = MSE_LOSS(y_true_values, y_preds)

        # Will continue until loss is less than stop loss (when the models stop progressing)
        while loss > stop_loss:
            for inputs, true_values in zip(data, y_true_values):

                # Variables that hold values we are going to need
                z1 = (self.w1 * inputs[0]) + (self.w2 * inputs[1]) +  self.b1
                h1 = sigmoid(z1)

                z2 = (self.w3 * inputs[0]) + (self.w4 * inputs[1]) + self.b2
                h2 = sigmoid(z2)

                z3 = (self.w5 * h1) + (self.w6 * h2) + self.b3
                o1 = sigmoid(z3)

                print(f"Input: {inputs}")
                print(f"True Value: {true_values}")
                print(f"Loss: {MSE_LOSS(true_values, o1)}")

                # Calculate partial derivatives
                D_L_Ypred = -2 * ( true_values - o1)

                # Pieces we need to take the partial derivatives
                D_Ypred_H1 = derive_sigmoid(z3) * self.w5
                D_H1_W1 = derive_sigmoid(z1) * inputs[0]
                D_H1_W2 = derive_sigmoid(z1) * inputs[1]
                D_H1_B1 = derive_sigmoid(z1)

                D_Ypred_H2 = derive_sigmoid(z3) * self.w6
                D_H2_W3 = derive_sigmoid(z2) * inputs[0]
                D_H2_W4 = derive_sigmoid(z2) * inputs[1]
                D_H2_B2 = derive_sigmoid(z2)

                D_Ypred_W5 = derive_sigmoid(z3) * h1
                D_Ypred_W6 = derive_sigmoid(z3) * h2
                D_Ypred_B3 = derive_sigmoid(z3)

                # Partial Derivatives with respect to each weight and bias
                D_L_W1 = D_L_Ypred * D_Ypred_H1 * D_H1_W1
                D_L_W2 = D_L_Ypred * D_Ypred_H1 * D_H1_W2
                D_L_W3 = D_L_Ypred * D_Ypred_H2 * D_H2_W3
                D_L_W4 = D_L_Ypred * D_Ypred_H2 * D_H2_W4
                D_L_W5 = D_L_Ypred * D_Ypred_W5
                D_L_W6 = D_L_Ypred * D_Ypred_W6

                DL_B1 = D_L_Ypred * D_Ypred_H1 * D_H1_B1
                DL_B2  = D_L_Ypred * D_Ypred_H2 * D_H2_B2
                DL_B3 = D_L_Ypred * D_Ypred_B3

                # How our weights our updated using SGD (W1 = W1 - nL/W1) where n is our learning rate
                self.w1 -= learn_rate * D_L_W1
                print(f"Updated W1: {self.w1}") # Outputting our updated weights

                self.w2 -= learn_rate * D_L_W2
                print(f"Updated W2: {self.w2}")  # Outputting our updated weights

                self.w3 -= learn_rate * D_L_W3
                print(f"Updated W3: {self.w3}") # Outputting our updated weights

                self.w4 -= learn_rate * D_L_W4
                print(f"Updated W4: {self.w4}") # Outputting our updated weights

                self.w5 -= learn_rate * D_L_W5
                print(f"Updated W5: {self.w5}") # Outputting our updated weights

                self.w6 -= learn_rate * D_L_W6
                print(f"Updated W6: {self.w6}") # Outputting our updated weights

                self.b1 -= learn_rate * DL_B1
                print(f"Updated B1: {self.b1}") # Outputting our updated bias

                self.b2 -= learn_rate * DL_B2
                print(f"Updated B2: {self.b2}") # Outputting our updated bias

                self.b3 -= learn_rate * DL_B3
                print(f"Updated B3:{self.b3}") # Outputting our updated bias

                epochs += 1 # Increase our epochs to see the number and divide by our number of samples (4) to get the real total of epochs

                # Updating per sample data set. This still works but instead of going through all the samples in the data set, it goes through each one and calculates each one. So the final correct final loss would be the last sample in the data set
                y_preds = np.apply_along_axis(self.feed_forward, 1, data) # Goes through one data set (Alice) calculates everything loss then moves to the next (Bob, then etc.) Calculates the loss per sample
                # apply_along_axis just applies the feedforward (with our current weights and bias) function to our dataset "data" by rows(Axis 1: rows, Axis 0: Columns) and puts them in a list.

                loss = MSE_LOSS(y_true_values, y_preds) # This just applies the MSE formula to give us the loss. So it subs each element by index, then exponentiates each element by the power of 2, then finds the mean.

                print(f"Predictions: {list(y_preds)}")
                print("Epoch %d loss: %.3f" % (epochs, loss))



def main():

    my_data = np.array([[-2,1]
                        ,[25,6],
                        [17,4],
                        [-15,-6]])
    true_values = np.array([1,0,0,1])

    my_height = float(input("How tall are you in inches? "))
    my_weight = float(input("How much do you weigh in pounds? "))

    network = OurNN()
    network.train(my_data, true_values) # Trains the data using the test data

    user_measurements = (converyHW_array(my_height, my_weight))

    if(network.feed_forward(user_measurements) > 0.5):
        print("You are female")
    if(network.feed_forward(user_measurements) < 0.5):
        print("You are male")

main()
