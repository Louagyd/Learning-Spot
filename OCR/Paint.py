
from PIL import Image, ImageDraw
import PIL
from tkinter import *
from tkinter.colorchooser import *
import os

class Paint:
    def __init__(self, scale_window= 5,  main_color = 'black', thickness = 1, rel_path = 'PaintedImages'):
        self.scale_window = scale_window
        self.width = 180*self.scale_window
        self.height = 60*self.scale_window
        self.thickness = thickness*self.scale_window
        self.center = self.height//2
        white = (255, 255, 255)
        self.main_color = main_color
        self.rel_path = rel_path

        self.root = Tk()
        Button(text='Select Color', command = self.getColor).grid(row = 0, column = 0)

        self.cv = Canvas(self.root, width=self.width, height=self.height, bg= 'white')
        self.cv.grid(row = 1, column = 0, columnspan = 6)

        self.image = PIL.Image.new("RGB", (self.width, self.height), white)
        self.draw = ImageDraw.Draw(self.image)

        self.cv.bind("<B1-Motion>", self.paint)

        self.button=Button(text="save",command=self.save).grid(row = 2, column = 4, sticky = W)
        self.save_result = Label()
        self.save_result.grid(row = 2, column = 5, sticky = W)
        Label(text="Target:").grid(row=2, column = 0, sticky=E)
        self.label = Text(height = 1, width = 7)
        self.label.grid(row = 2, column = 1, sticky = W)
        self.root.mainloop()



    def save(self):
        num_saved_images = os.listdir('PaintedImages').__len__() - 1
        img = self.image.resize((self.width//self.scale_window,self.height//self.scale_window))
        img.save(self.rel_path + '/' + str(num_saved_images) + '.png')
        with open("PaintedImages/labels.txt", "a") as labelsfile:
            labelsfile.write(str(self.label.get("1.0", END)))
        self.save_result.config(text = 'Saved as /' + str(num_saved_images) + '.png')
    def paint(self, event):
        x1, y1 = (event.x - self.thickness), (event.y - self.thickness)
        x2, y2 = (event.x + self.thickness), (event.y + self.thickness)
        self.cv.create_oval(x1, y1, x2, y2, fill = self.main_color, outline = self.main_color)
        self.draw.ellipse([x1, y1, x2, y2], fill = self.main_color)

    def getColor(self):
        color = askcolor()
        self.main_color = color[1]

drawing = Paint(thickness=3)


