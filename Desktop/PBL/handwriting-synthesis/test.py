import numpy as np
# import sys
from handwriting_synthesis import Hand



text = """Fuck a pigeonhole, I'm a night owl, this a different mode
I might have to make her paint a 6 on her pinky toe
Heard you with a shooting guard, just let a nigga know
I would have you courtside, not the middle row
All good, love, in a minute, though
I can't stress about no bitch 'cause I'm a timid soul
Plus I'm cookin' up ambition on a kitchen stove
Pot start to bubble, see the suds, that shit good to go
"""

lines = text.split('\n')

if __name__ == '__main__':
    hand =Hand()

    biases = [.75 for i in lines]
    styles = [9 for i in lines]
    stroke_colors = ['red', 'green', 'black', 'blue', 'red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2, 3 , 2, 3, 2]

    hand.write(
        filename='img/jimmy_cooks.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths
    )

    #input_svg_file = "img/jimmy_cooks.svg"  # Path to existing SVG file
    #output_svg_file = "resized_output.svg"  # Path to resized SVG file

    #svg_resizer = hand.resize_svg(input_svg_file, output_svg_file)