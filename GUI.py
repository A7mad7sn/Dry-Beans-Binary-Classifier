import tkinter as tk
from tkinter import ttk
from PIL import ImageTk


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dry Beans Classifier - NN -Task1")
        center_window(self.root, 500, 300)

        
        self.root.resizable(False, False)

        self.backgroundimg = ImageTk.PhotoImage(file="Background.jpg")

        canvas = tk.Canvas(self.root)
        canvas.create_image(0, 0, image=self.backgroundimg, anchor=tk.NW)
        canvas.create_text(64, 24, text="Select Your 2 Features:", font=("times", 9, "bold"))
        canvas.create_text(61, 55, text="Select Your 2 Classes:", font=("times", 9, "bold"))
        canvas.create_text(42, 85, text="Learning Rate:", font=("times", 9, "bold"))
        canvas.create_text(54, 115, text="Number Of Epochs:", font=("times", 9, "bold"))
        canvas.create_text(44, 144, text="MSE threshold:", font=("times", 9, "bold"))
        canvas.create_text(33, 172, text="Algorithm:", fill="white", font=("times", 9, "bold"))
        canvas.pack(fill="both", expand=True)

        s = ttk.Style()
        s.configure('Wild.TRadiobutton', background="black", foreground='white')

        def second_combo1(*args):
            if self.combo1.get() == "Area":
                self.combo2.config(values=["Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
            elif self.combo1.get() == "Perimeter":
                self.combo2.config(values=["Area", "MajorAxisLength", "MinorAxisLength", "roundnes"])
            elif self.combo1.get() == "MajorAxisLength":
                self.combo2.config(values=["Area", "Perimeter", "MinorAxisLength", "roundnes"])
            elif self.combo1.get() == "MinorAxisLength":
                self.combo2.config(values=["Area", "Perimeter", "MajorAxisLength", "roundnes"])
            else:
                self.combo2.config(values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"])
            if self.combo2.get() == "Area":
                self.combo1.config(values=["Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
            elif self.combo2.get() == "Perimeter":
                self.combo1.config(values=["Area", "MajorAxisLength", "MinorAxisLength", "roundnes"])
            elif self.combo2.get() == "MajorAxisLength":
                self.combo1.config(values=["Area", "Perimeter", "MinorAxisLength", "roundnes"])
            elif self.combo2.get() == "MinorAxisLength":
                self.combo1.config(values=["Area", "Perimeter", "MajorAxisLength", "roundnes"])
            else:
                self.combo1.config(values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength"])


        var1 = tk.StringVar()
        self.combo1 = ttk.Combobox(state="readonly", textvariable=var1,
                                   values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
        self.combo1.place(x=130, y=15)
        self.combo1.bind("<<ComboboxSelected>>", second_combo1)

        self.combo2 = ttk.Combobox(state="readonly",
                                   values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
        self.combo2.place(x=280, y=15)

        def second_combo2(*args):
            if self.combo3.get() == "BOMBAY":
                self.combo4.config(values=["CALI", "SIRA"])
            elif self.combo3.get() == "CALI":
                self.combo4.config(values=["BOMBAY", "SIRA"])
            else:
                self.combo4.config(values=["BOMBAY", "CALI"])
            if self.combo4.get() == "BOMBAY":
                self.combo3.config(values=["CALI", "SIRA"])
            elif self.combo4.get() == "CALI":
                self.combo3.config(values=["BOMBAY", "SIRA"])
            else:
                self.combo3.config(values=["BOMBAY", "CALI"])

        var2 = tk.StringVar()
        self.combo3 = ttk.Combobox(state="readonly", textvariable=var2, values=["BOMBAY", "CALI", "SIRA"])
        self.combo3.place(x=130, y=45)
        self.combo3.bind("<<ComboboxSelected>>", second_combo2)

        self.combo4 = ttk.Combobox(state="readonly", values=["BOMBAY", "CALI", "SIRA"])
        self.combo4.place(x=280, y=45)

        self.learning_rate_entry = tk.Entry(self.root, width=23)
        self.learning_rate_entry.place(x=130, y=75)

        self.epochs_entry = tk.Entry(self.root, width=23)
        self.epochs_entry.place(x=130, y=105)

        self.mse_entry = tk.Entry(self.root, width=23)
        self.mse_entry.place(x=130, y=135)

        self.varCheck = tk.IntVar()

        self.check_box = tk.Checkbutton(self.root, text="Use Bias?", variable=self.varCheck, onvalue=1, offvalue=0,background='gold')
        self.check_box.place(x=4, y=210)

        self.var = tk.StringVar()

        self.r1 = ttk.Radiobutton(self.root, text="Perceptron", variable=self.var, value="perceptron", style="Wild.TRadiobutton")
        self.r1.place(x=4, y=185)

        self.r2 = ttk.Radiobutton(self.root, text="Adaline", variable=self.var, value="adaline", style="Wild.TRadiobutton")
        self.r2.place(x=100, y=185)

        self.var.set('perceptron')

        self.enter_button = tk.Button(self.root, text="Train", height=2, width=15, bg="#5DADE2", command=self.get_inputs)
        self.enter_button.place(x=200, y=250)

        self.Inputs = []

        self.root.mainloop()

        
        
    def get_inputs(self):
        
        if(self.combo1.get() == '' or 
           self.combo2.get() == '' or 
           self.combo3.get() == '' or 
           self.combo4.get() == '' or 
           self.learning_rate_entry.get() == '' or 
           self.epochs_entry.get() == '' or 
           self.mse_entry.get() == '' 
           ):
            tk.messagebox.showinfo(title='Invalid',message='Please fill all required entries')
            return
                
        selected_feature = []
        features_indx = []
        selected_classes = []
        selected_feature.append(self.combo1.get())
        selected_feature.append(self.combo2.get())
        
        if self.combo1.get() == "Area":
            features_indx.append(0)
        elif self.combo1.get() == "Perimeter":
            features_indx.append(1)
        elif self.combo1.get() == "MajorAxisLength":
            features_indx.append(2)
        elif self.combo1.get() == "MinorAxisLength":
            features_indx.append(3)
        elif self.combo1.get() == "roundnes":
            features_indx.append(4)
            
        if self.combo2.get() == "Area":
            features_indx.append(0)
        elif self.combo2.get() == "Perimeter":
            features_indx.append(1)
        elif self.combo2.get() == "MajorAxisLength":
            features_indx.append(2)
        elif self.combo2.get() == "MinorAxisLength":
            features_indx.append(3)
        elif self.combo2.get() == "roundnes":
            features_indx.append(4)
           
            
        if self.combo3.get() == "BOMBAY":
            selected_classes.append(0)
        elif self.combo3.get() == "CALI":
            selected_classes.append(1)
        else:
            selected_classes.append(2)

        if self.combo4.get() == "BOMBAY":
            selected_classes.append(0)
        elif self.combo4.get() == "CALI":
            selected_classes.append(1)
        else:
            selected_classes.append(2)

        learning_rate = float(self.learning_rate_entry.get())

        epochs_num = int(self.epochs_entry.get())

        mse = float(self.mse_entry.get())

        if self.varCheck.get() == 1:
            addBiasOrNot = True
        else:
            addBiasOrNot = False

        if self.var.get() == "perceptron":
            name_of_chosen_algorithm = "Single Layer Perceptron"
        else:
            name_of_chosen_algorithm = "Adaline"
            
        all_inputs = [selected_feature, selected_classes, learning_rate, epochs_num, mse, addBiasOrNot, name_of_chosen_algorithm,features_indx]
        
        self.Inputs = all_inputs
        
        self.root.destroy()
        
        del self   
        

        
        
