import tkinter as tk
from tkinter import colorchooser
import numpy as np

DEFAULT_COLOR='#ffffff'

class BraceletSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Bracelet Pattern Designer")

        self.rows = 2
        self.threads = 6
        self.diamond_size = 40
        self.diamonds = {}

        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        self.controls = tk.Frame(self.root)
        self.controls.pack(side='bottom')
        self.info = tk.Frame(self.root)
        self.info.pack(side='top')
        
        num_colors=self.get_num_colors() # why returning 0 initially???
        self.dimensions_label=tk.Label(self.info, text=f"Threads x Rows = {self.threads} x {self.rows}, {num_colors} colors")
        self.dimensions_label.pack(side="bottom")
        tk.Button(self.controls, text="Add Row", command=self.add_row).pack(side="left")
        tk.Button(self.controls, text="Add Thread", command=self.add_column).pack(side="left")
        tk.Button(self.controls, text="Clear", command=self.clear_colors).pack(side="left")

        self.draw_grid()

    def update_dimensions_label(self):
        num_colors=self.get_num_colors()
        self.dimensions_label.config(text=f"Threads x Rows = {self.threads} x {self.rows}, {num_colors} colors")

    def get_num_colors(self):
        return len(set(self.diamonds.values()))

    def draw_grid(self):
        self.canvas.delete("all")
        self.diamonds.clear()
        for row in range(self.rows):
            cols=int(np.floor(self.threads/2))
            if(self.threads%2==0 and row%2!=0):
                cols-=1
            for col in range(cols):
                self.draw_diamond(row, col)

    def draw_diamond(self, row, col, fill_color=DEFAULT_COLOR):
        size = self.diamond_size
        x = col * size + size
        y = row * size/2 + size

        if(row%2!=0): # odd rows are shifted
            x+=int(size/2)

        points = [
            x, y - size // 2,
            x + size // 2, y,
            x, y + size // 2,
            x - size // 2, y
        ]

        item = self.canvas.create_polygon(points, fill=fill_color, outline="black")
        self.canvas.tag_bind(item, "<Button-1>", lambda e, i=item: self.change_color(i))
        self.diamonds[item] = fill_color

    def change_color(self, item):
        color = colorchooser.askcolor(title="Pick a color")[1]
        if color:
            self.canvas.itemconfig(item, fill=color)
            self.diamonds[item] = color
        num_colors=self.get_num_colors()
        self.dimensions_label.config(text=f"Threads x Rows = {self.threads} x {self.rows}, {num_colors} colors")

    def add_row(self):
        self.rows += 1
        self.draw_grid()
        self.update_dimensions_label()

    def add_column(self):
        self.threads += 1
        self.draw_grid()
        self.update_dimensions_label()

    def clear_colors(self):
        for item in self.diamonds:
            self.canvas.itemconfig(item, fill=DEFAULT_COLOR)
            self.diamonds[item] = DEFAULT_COLOR

if __name__ == "__main__":
    root = tk.Tk()
    app = BraceletSolver(root)
    root.mainloop()
