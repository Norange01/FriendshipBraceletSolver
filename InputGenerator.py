from itertools import product
import random
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from IPython.display import display

#############################
# For testing
#############################
max_iterations=10000

OVAL_RADIUS=20
LINE_THICKNESS=10
KNOT_HORIZ_SPACING=3*OVAL_RADIUS
KNOT_VERT_SPACING=int(KNOT_HORIZ_SPACING/2)-(OVAL_RADIUS)
HORIZ_MARGIN=80
VERT_MARGIN=40
NONE_VALUE='#808080'

def get_knot_result(arrow_index, left_color, right_color):
    knot_color=None
    left_out_color=None
    right_out_color=None

    if(arrow_index==0 or arrow_index==2): # fwd or fwdbwd
        knot_color=left_color
    else: # bwd or bwdfwd
        knot_color=right_color


    if(arrow_index==0 or arrow_index==1): # fwd or bwd
        left_out_color=right_color
        right_out_color=left_color
    else: # fwdbwd or bwdfwrd
        left_out_color=left_color
        right_out_color=right_color

    return knot_color, left_out_color, right_out_color


def display_target_design(target_design):
    num_rows=len(target_design)
    num_threads=len(target_design[0])*2
    if(len(target_design[0])==len(target_design[1])):
        num_threads+=1
    num_cols=int(np.floor(float(num_threads)/2))
    
    # Initializing the base image
    image_width=HORIZ_MARGIN*2+(num_cols*OVAL_RADIUS*2)+((num_cols-1)*KNOT_HORIZ_SPACING)
    image_height=VERT_MARGIN*2+(num_rows*OVAL_RADIUS*2)+((num_rows-1)*KNOT_VERT_SPACING)

    image = Image.new('RGBA',(image_width, image_height))

    draw = ImageDraw.Draw(image)

    for row in range(num_rows):
        for col in range(num_cols):
            if(row%2==0): # if row is even
                oval_x_start=int(HORIZ_MARGIN+(2*col*OVAL_RADIUS)+(col*KNOT_HORIZ_SPACING))
            else: # if row is odd
                oval_x_start=int((OVAL_RADIUS)+(KNOT_HORIZ_SPACING/2)+HORIZ_MARGIN+(2*col*OVAL_RADIUS)+(col*KNOT_HORIZ_SPACING))
                if(num_threads%2==0 and col==num_cols-1): # if num of threads is even and we reached end of row
                    continue # no knot
            
            oval_y_start=int(VERT_MARGIN+(2*row*OVAL_RADIUS)+(row*KNOT_VERT_SPACING))
            
            color_hex = target_design[row][col]

            draw.ellipse((oval_x_start, oval_y_start, oval_x_start+OVAL_RADIUS*2, oval_y_start+OVAL_RADIUS*2), fill = color_hex, outline='black')

    image=ImageOps.expand(image, border=1, fill='white')
    display(image)

def random_target_design_generator(num_threads, num_rows, num_colors, display=False):
    num_cols=int(np.floor(float(num_threads)/2))
    symmetric=num_threads%2!=0

    if(num_rows%2!=0):
        raise Exception("Number of rows must be even.")
    
    if(num_colors>num_threads):
        raise Exception("Number of colors must be less than or equal to number of threads.")

    if(num_colors>10):
        raise Exception("Maximum number of colors is 10.")
    
    for i in range(max_iterations):
        # generate random knots array
        knots_arr=[]
        for i in range(num_rows):
            knots_row=[]
            row_num_cols=num_cols
            if(i%2!=0 and not symmetric):
                row_num_cols-=1
            for j in range(row_num_cols):
                knots_row.append(random.randint(0,3))
            knots_arr.append(knots_row)

        # choosing colors
        base_colors=['#ff0000', '#ffff00', '#0000ff', '#2e2e2e', '#ffffff','#ff8c00', '#65c92a', '#00ddff','#8800ff','#ffabf2']

        # generate random initial threads
        initial_threads=[]
        for i in range(num_colors):
            initial_threads.append(base_colors[i])
        
        while(len(initial_threads)<num_threads):
            initial_threads.append(base_colors[random.randint(0,num_colors-1)])

        random.shuffle(initial_threads)

        threads_arr=[initial_threads]

        # apply the random knots to generate a target design
        target_design=[]
        for row in range(num_rows):
            target_design_row=[]
            threads_row=threads_arr[-1]
            new_threads_row=[]
            if(row%2!=0): # if row is odd
                new_threads_row.append(threads_row[0])

            for col in range(num_cols):
                if(row%2==0): # if row is even
                    left_in=threads_row[col*2]
                    right_in=threads_row[(col*2)+1]
                else: # if row is odd
                    if(num_threads%2==0 and col==num_cols-1): # if num of threads is even and we reached end of row
                        new_threads_row.append(threads_row[-1])
                        continue # no knot
                    left_in=threads_row[(col*2)+1]
                    right_in=threads_row[(col*2)+2]

                color_hex, left_out, right_out =get_knot_result(knots_arr[row][col],left_in, right_in)
                target_design_row.append(color_hex)

                new_threads_row.append(left_out)
                new_threads_row.append(right_out)
            if(num_threads%2!=0 and row%2==0):
                new_threads_row.append(threads_row[-1])
            threads_arr.append(new_threads_row)
            target_design.append(target_design_row)

        if(threads_arr[-1]!=threads_arr[0]): # if rule 5 was violated
            continue

        if(display==True):
            display_target_design(target_design)
            #display_diagram(knots_arr, threads_arr)
        return target_design
    # if loop was over and no target design found
    raise Exception("Could not find a design that doesn't violate Rule 5.")