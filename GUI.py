import tkinter as tk
from tkinter import colorchooser
import numpy as np

DEFAULT_COLOR='#ffffff'
MIN_ROWS=2
MIN_THREADS=6
MAX_ROWS=40
MAX_THREADS=30


class BraceletSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Bracelet Pattern Designer")

        self.rows = MIN_ROWS
        self.threads = MIN_THREADS
        self.diamond_size = 40
        self.diamonds = {}
        self.diamond_positions={}

        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        self.controls = tk.Frame(self.root)
        self.controls.pack(side='bottom')
        self.info = tk.Frame(self.root)
        self.info.pack(side='top')
        
        num_colors=self.get_num_colors()
        self.dimensions_label=tk.Label(self.info, text=f"Threads x Rows = {self.threads} x {self.rows}, {num_colors} colors")
        self.dimensions_label.pack(side="bottom")
        tk.Button(self.controls, text="Add Row Pair", command=self.add_two_rows).pack(side="left")
        tk.Button(self.controls, text="Add Thread", command=self.add_thread).pack(side="left")
        tk.Button(self.controls, text="Remove Thread", command=self.remove_thread).pack(side="left")
        tk.Button(self.controls, text="Clear", command=self.clear_colors).pack(side="left")
        tk.Button(self.controls, text="Remove Row Pair", command=self.remove_two_rows).pack(side="left")
        tk.Button(self.controls, text="SOLVE", command=self.solve).pack(side="left")

        self.draw_grid()
        self.update_dimensions_label()

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
        self.diamond_positions[item] = (row, col)

    def change_color(self, item):
        color = colorchooser.askcolor(title="Pick a color")[1]
        if color:
            self.canvas.itemconfig(item, fill=color)
            self.diamonds[item] = color
        self.update_dimensions_label()

    def add_row(self):
        max_bound=int(np.floor(self.threads/2))
        if(self.threads%2==0 and self.rows%2!=0):
            max_bound-=1
        for i in range(max_bound):
            self.draw_diamond(row=self.rows, col=i)
        self.rows += 1
        self.update_dimensions_label()

    def remove_row(self):
        # find IDs of diamonds
        keys=list(self.diamond_positions.keys())
        for id in keys:
            if self.diamond_positions[id][0]==self.rows-1:
                self.canvas.delete(id)
                del self.diamond_positions[id]
        self.rows -= 1
        self.update_dimensions_label()

    def remove_two_rows(self):
        if(self.rows-2>=MIN_ROWS):
            self.remove_row()
            self.remove_row()
    
    def add_two_rows(self):
        if(self.rows+2<=MAX_ROWS):
            self.add_row()
            self.add_row()

    def add_thread(self):
        if(self.threads+1<=MAX_THREADS):
            column=int(np.floor(self.threads-1)/2)
            for i in range(self.rows):
                if((i%2==0 and self.threads%2!=0) or (i%2!=0 and self.threads%2==0)):
                    self.draw_diamond(row=i, col=column)
            self.threads += 1
            self.update_dimensions_label()

    def remove_thread(self):
        if(self.threads-1>=MIN_THREADS):
            column=int(np.floor(self.threads)/2)-1
            keys=list(self.diamond_positions.keys())
            for id in keys:
                if (self.diamond_positions[id][1]==column and ((self.diamond_positions[id][0]%2==0 and self.threads%2==0) or (self.diamond_positions[id][0]%2!=0 and self.threads%2!=0))):
                    self.canvas.delete(id)
                    del self.diamond_positions[id]
            self.threads-=1
            self.update_dimensions_label()

    def clear_colors(self):
        for item in self.diamonds:
            self.canvas.itemconfig(item, fill=DEFAULT_COLOR)
            self.diamonds[item] = DEFAULT_COLOR

    def solve(self):
        target_design = []
        for row_pair in range(int(self.rows/2)):
            target_design.append(['#ffffff']*int(np.floor(self.threads/2)))
            if(self.threads%2==0):
                target_design.append(['#ffffff']*int(np.floor(self.threads/2)-1))
            else:
                target_design.append(['#ffffff']*int(np.floor(self.threads/2)))
        for i in range(len(self.diamonds)):
            target_design[self.diamond_positions[i+1][0]][self.diamond_positions[i+1][1]]=self.diamonds[i+1]

        
        


if __name__ == "__main__":
    root = tk.Tk()
    app = BraceletSolver(root)
    root.mainloop()
