import tkinter as tk
from tkinter import colorchooser
import numpy as np
from Solver import Solver
from PIL import Image, ImageTk, ImageDraw, ImageOps
import time
import imageio
from IPython.display import display

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
        num_colors=self.get_num_colors()
        self.diagrams = []

        self.controls = tk.Frame(self.root)
        self.controls.pack(side='top')
        self.info = tk.Frame(self.root)
        self.info.pack(side='top')
        self.outputOptions = tk.Frame(self.root)
        self.outputOptions.pack(side='bottom')

        self.dimensions_label=tk.Label(self.info, text=f"Threads x Rows = {self.threads} x {self.rows}, {num_colors} colors")
        self.dimensions_label.pack(side="bottom")
        
        self.addRowPairBtn = tk.Button(self.controls, text="Add Row Pair", command=self.add_two_rows)
        self.addRowPairBtn.pack(side="left")
        self.removeRowPairBtn = tk.Button(self.controls, text="Remove Row Pair", command=self.remove_two_rows, state="disabled")
        self.removeRowPairBtn.pack(side="left")

        self.addThreadBtn = tk.Button(self.controls, text="Add Thread", command=self.add_thread)
        self.addThreadBtn.pack(side="left")
        self.removeThreadBtn=tk.Button(self.controls, text="Remove Thread", command=self.remove_thread, state="disabled")
        self.removeThreadBtn.pack(side="left")

        tk.Button(self.controls, text="Clear", command=self.clear_colors).pack(side="left")
        
        tk.Button(self.controls, text="SOLVE", command=self.solve).pack(side="left")

        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        self.replayBtn=tk.Button(self.outputOptions, text="Replay", command=self.replay_animation, state="disabled")
        self.replayBtn.pack(side="left")
        self.savePNGBtn=tk.Button(self.outputOptions, text="Save PNG", command=self.savePNG, state="disabled")
        self.savePNGBtn.pack(side="left")
        self.saveGIFBtn=tk.Button(self.outputOptions, text="Save GIF", command=self.saveGIF, state="disabled")
        self.saveGIFBtn.pack(side="left")

        self.img_space = tk.Label(self.root, image="")
        self.img_space.pack()
        self.current_frame=0
        self.photoimage_objects=[]

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
        print("grid drawn")
        print(self.diamonds)
        print(self.diamond_positions)

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
        print("color changed")
        print(self.diamonds)
        print(self.diamond_positions)

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
            print("removed two rows")
            print(self.diamonds)
            print(self.diamond_positions)
            if(self.rows<=MIN_ROWS):
                self.removeRowPairBtn.configure(state="disabled")
            elif(self.rows<MAX_ROWS):
                self.addRowPairBtn.configure(state="normal")

    
    def add_two_rows(self):
        if(self.rows+2<=MAX_ROWS):
            self.add_row()
            self.add_row()
            print("added two rows")
            print(self.diamonds)
            print(self.diamond_positions)
            if(self.rows>=MAX_ROWS):
                self.addRowPairBtn.configure(state="disabled")
            elif(self.rows>MIN_ROWS):
                self.removeRowPairBtn.configure(state="normal")

    def add_thread(self):
        if(self.threads+1<=MAX_THREADS):
            column=int(np.floor(self.threads-1)/2)
            for i in range(self.rows):
                if((i%2==0 and self.threads%2!=0) or (i%2!=0 and self.threads%2==0)):
                    self.draw_diamond(row=i, col=column)
            self.threads += 1
            self.update_dimensions_label()
            print("added thread")
            print(self.diamonds)
            print(self.diamond_positions)
            if(self.threads>=MAX_THREADS):
                self.addThreadBtn.configure(state="disabled")
            elif(self.threads>MIN_THREADS):
                self.removeThreadBtn.configure(state="normal")

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
            print("removed thread")
            print(self.diamonds)
            print(self.diamond_positions)
            if(self.threads<=MIN_THREADS):
                self.removeThreadBtn.configure(state="disabled")
            elif(self.threads<MAX_THREADS):
                self.addThreadBtn.configure(state="normal")

    def clear_colors(self):
        for item in self.diamonds:
            self.canvas.itemconfig(item, fill=DEFAULT_COLOR)
            self.diamonds[item] = DEFAULT_COLOR
        

    def solve(self):
        print(self.diamonds)
        print(self.diamond_positions)
        target_design = []
        for row_pair in range(int(self.rows/2)):
            target_design.append(['#ffffff']*int(np.floor(self.threads/2)))
            if(self.threads%2==0):
                target_design.append(['#ffffff']*int(np.floor(self.threads/2)-1))
            else:
                target_design.append(['#ffffff']*int(np.floor(self.threads/2)))
        '''for i in range(len(self.diamonds)):
            target_design[self.diamond_positions[i+1][0]][self.diamond_positions[i+1][1]]=self.diamonds[i+1]'''

        for item_id, (row, col) in self.diamond_positions.items():
            target_design[row][col] = self.diamonds[item_id]


        solver=Solver(target_design)
        solution=solver.solve(False,0)
        self.diagrams = solver.get_solution_diagrams()
        

        frames=len(self.diagrams)
        self.photoimage_objects=[]
        for i in range(frames):
            obj = ImageTk.PhotoImage(self.diagrams[i])
            self.photoimage_objects.append(obj)

        self.activateOutputButtons()
        
        #gif_file = Image.open("solution.gif")
        
        self.current_frame=0
        self.animate_gif()


    def animate_gif(self):
        if(self.current_frame<len(self.photoimage_objects)):
            self.img_space.configure(image=self.photoimage_objects[self.current_frame])
            self.current_frame+=1
            self.root.after(100, self.animate_gif)

    def savePNG(self):
        self.diagrams[-1].save("FriendshipBracelet_Solution.png")
        

    def saveGIF(self):
        if not self.diagrams:
            print("No diagrams to save.")
            return
        imageio.mimsave("FriendshipBracelet_Solution.gif", self.diagrams, loop=0, disposal=2)
        print("GIF saved successfully.")


    def replay_animation(self):
        self.current_frame=0
        self.animate_gif()

    def activateOutputButtons(self):
        self.replayBtn.configure(state="normal")
        self.saveGIFBtn.configure(state="normal")
        self.savePNGBtn.configure(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = BraceletSolver(root)
    root.mainloop()
