from preprocessing import preprocessing,data_extraction_and_splitting
from perceptron import Perceptron
from Adaline import Adaline
from GUI import GUI
from Evaluation import Evaluator


if __name__ == "__main__":
    
    # Getting user inputs:
    gui = GUI()
    features, classes, lr, epochs, mse_threshold, addBias, chosen_algorithm ,feature_index_list = gui.Inputs
    
    # Preprocessing only the selected features:
    data_set,scaler = preprocessing(features)
    
    # Getting only the required classes then splitting:
    x_train,x_test,y_train,y_test = data_extraction_and_splitting(data_set,features,classes)
    
    # as initialization
    model = None
    
    if(chosen_algorithm=='Single Layer Perceptron'):
        model = Perceptron(learning_rate=lr,epochs=epochs,addBias=addBias,mse_threshold=mse_threshold)
                
    elif(chosen_algorithm=='Adaline'):
        model = Adaline(learning_rate=lr,epochs=epochs,addBias=addBias,mse_threshold=mse_threshold)

    # Train the chosen NN:
    model.train(x_train,y_train)
    
    # Evaluating & Testing:
    Evaluator(model,x_test,y_test,feature_index_list,classes,scaler,chosen_algorithm)
    
