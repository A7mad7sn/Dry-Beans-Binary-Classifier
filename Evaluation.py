import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk
from GUI import center_window


class Evaluator:

    def __init__(self, model, x_test, y_test, feature_indx, classes_indx,scaler,algorithm):
        self.all_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
        self.all_classes = ['BOMBAY', 'CALI', 'SIRA']
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.y_pred = self.model.test(x_test)
        self.feature1 = self.all_features[feature_indx[0]]
        self.feature2 = self.all_features[feature_indx[1]]
        self.class1 = self.all_classes[classes_indx[0]]
        self.class2 = self.all_classes[classes_indx[1]]
        self.TP, self.TN, self.FP, self.FN = [0, 0, 0, 0]
        self.accuarcy_calculator(self.y_test, self.y_pred)
        self.scaler = scaler
        self.algorithm = algorithm
        self.gui()
        
        
    def gui(self):
        root = tk.Tk()
        root.title("Classifier Evaluation")
        root.geometry("976x580")
        center_window(root,1094,650)
        
        self.backgroundimg = ImageTk.PhotoImage(file="backgroundforclassifier.jpg")

        canvas = tk.Canvas(root)
        canvas.create_image(0, 0, image=self.backgroundimg, anchor=tk.NW)
        canvas.pack(fill="both", expand=True)
        
        algorithm_label =tk.Label(canvas,text=self.algorithm,background='gold',font=("times", 30, "bold"))
        algorithm_label.pack(padx=10,pady=10)


        self.data_visualization(canvas)
        
        
        self.calculate_confusion_matrix(canvas)
        
        accuracy_frame = tk.Frame(canvas,background='gold')
        accuracy_frame.pack(padx=10, pady=10 ,expand=True,fill='both',)
        accuracy_label = tk.Label(accuracy_frame, text=f"Accuracy: {self.accuarcy} %",background='gold',font=("times", 15, "bold"))
        accuracy_label.pack(expand=True, fill='both')
        
        getter_frame = tk.Frame(accuracy_frame,background='gold')
        global feature1_tbox
        global feature2_tbox
        global feature1_norm
        global feature2_norm
        feature1_label = tk.Label(getter_frame,text=self.feature1,background='gold',font=("times", 12, "bold"))
        feature2_label = tk.Label(getter_frame,text=self.feature2,background='gold',font=("times", 12, "bold"))
        feature1_tbox = tk.Entry(getter_frame)
        feature2_tbox = tk.Entry(getter_frame)
        feature1_norm = tk.Label(getter_frame,background='gold',font=("times", 12, "bold"))
        feature2_norm = tk.Label(getter_frame,background='gold',font=("times", 12, "bold"))
        feature1_label.grid(row=0,column=0)
        feature1_tbox.grid(row=0,column=1)
        feature2_label.grid(row=1,column=0)
        feature2_tbox.grid(row=1,column=1)
        getter_frame.pack()
        
        predict_button = tk.Button(accuracy_frame, text="Predict", command=self.predict_sample, height=2, bg="#5DADE2",width=580,font=("times", 15, "bold"))
        predict_button.pack(padx=10,pady=10,side=tk.BOTTOM)

        root.mainloop()
    
    def data_visualization(self,root):
        plot_frame = tk.Frame(root)
        plot_frame.pack(fill='both',expand=True,padx=10, pady=10)

        unique_values = np.unique(self.y_pred)
        num_unique = len(unique_values)
        cmap = ListedColormap(plt.cm.get_cmap('cool')(np.linspace(0, 1, num_unique)))
        plt.scatter(self.x_test.T[0], self.x_test.T[1], c=self.y_pred, cmap=cmap)
        line_x1_points = np.arange(min(self.x_test.T[0])-0.1, max(self.x_test.T[0])+0.1, 0.1)
        plt.plot(line_x1_points, self.get_x2(line_x1_points), label='Decision Boundary')
        colorbar = plt.colorbar(ticks=unique_values)
        if(num_unique==np.array([1])):
            colorbar.set_ticklabels([self.class2])
        elif(num_unique==np.array([-1])):
            colorbar.set_ticklabels([self.class1])
        else:
            colorbar.set_ticklabels([self.class2, self.class1])
        plt.xlabel(self.feature1)
        plt.ylabel(self.feature2)
        plt.legend()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
        canvas.get_tk_widget().pack(fill='both',expand=True)


    def calculate_confusion_matrix(self, root):
        confusion_frame = tk.Frame(root)
        confusion_frame.pack(padx=10,pady=10,side=tk.RIGHT,fill='both')


        plt.figure()
        plt.imshow([[self.TP, self.FP], [self.FN, self.TN]], cmap='Blues', interpolation='nearest')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str([[self.TP, self.FN], [self.FP, self.TN]][i][j]), horizontalalignment='center',
                         verticalalignment='center', fontsize=14, color='red')

        plt.xticks([0, 1], ["Positive", "Negative"])
        plt.yticks([0, 1], ["Positive","Negative"],rotation='vertical',ha='right', va='center')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        canvas = FigureCanvasTkAgg(plt.gcf(), master=confusion_frame)
        canvas.get_tk_widget().pack()
   

    def predict_sample(self):
        global feature1_tbox
        global feature2_tbox
        global feature1_norm
        global feature2_norm
        if(feature1_tbox.get()=='' or feature2_tbox.get() == ''):
            tk.messagebox.showinfo(title='Invalid',message='Please fill all required entries')
            return
        feature1 = float(feature1_tbox.get())
        feature2 = float(feature2_tbox.get())

        sample = np.array([[feature1,feature2]])
        sample = self.scaler.transform(sample)
        
        feature1 = sample[0, 0]  
        feature2 = sample[0, 1]  
        
        feature1_norm.configure(text= f'Norm:({feature1})')
        feature2_norm.configure(text= f'Norm:({feature2})')
        feature1_norm.grid(row=0,column=2)
        feature2_norm.grid(row=1,column=2)
        

        y_pred = self.model.test(sample)
        if y_pred[0] == 1:
            predicted_label = self.class1
        elif y_pred[0] == -1 :
            predicted_label = self.class2

        tk.messagebox.showinfo("Prediction Result", f"Predicted class: {predicted_label}")

    def accuarcy_calculator(self,y_real,y_pred):
        for i in range(len(y_real)):
            if(y_real[i] == 1) and (y_pred[i] == 1):
                self.TP = self.TP + 1
            elif(y_real[i] == -1) and (y_pred[i] == -1):
                self.TN = self.TN + 1
            elif(y_real[i] == -1) and (y_pred[i] == 1):
                self.FP = self.FP + 1
            elif(y_real[i] == 1) and (y_pred[i] == -1):
                self.FN = self.FN + 1
        self.accuarcy = 100 * (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP)

    def get_x2(self,x1):
        return -(self.model.bias+self.model.weights[0]*x1)/self.model.weights[1]