import logging
import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.hand._draw import _draw
from handwriting_synthesis.rnn import RNN

import svgwrite
from svgwrite.container import Group
from svgwrite.shapes import Rect, Circle, Ellipse, Polygon, Polyline, Line
from svgwrite.text import Text, TSpan, TRef


class Hand(object):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = RNN(
            log_dir='logs',
            checkpoint_dir=checkpoint_path,
            prediction_dir=prediction_path,
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 60:
                raise ValueError(
                    (
                        "Each line must be at most 60 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        _draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f"{style_path}/style-{style}-strokes.npy")
                c_p = np.load(f"{style_path}/style-{style}-chars.npy").tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples
    
    #def resize_svg(self, input_svg, output_svg):
        '''#drawing = svgwrite.Drawing(filename=input_svg)

        ## Original dimensions of the SVG
        #original_width = 1000
        #original_height = 600

        # Dimensions of A4 paper in millimeters
        #a4_width_mm = 210
        #a4_height_mm = 297

        # Calculate scaling factors for width and height
       # scale_factor_width = a4_width_mm / original_width
       # scale_factor_height = a4_height_mm / original_height

        # Choose the smaller scaling factor to ensure the entire SVG fits within A4 size
       # scale_factor = min(scale_factor_width, scale_factor_height)

        # Calculate new dimensions based on the chosen scaling factor
       # new_width = original_width * scale_factor
       # new_height = original_height * scale_factor

        # Update width and height attributes of the drawing
       # drawing.attribs['width'] = f"{new_width}px"
       # drawing.attribs['height'] = f"{new_height}px"

        # Apply the same scale factor to the transformation in <defs> if it exists
      #  if 'transform' in drawing.defs.attribs:
      #      transform_value = drawing.defs.attribs['transform']
      #      scale_values = transform_value.replace('scale(', '').replace(')', '').split(',')
      #      new_scale_x = float(scale_values[0]) * scale_factor
      #      new_scale_y = float(scale_values[1]) * scale_factor
      #      drawing.defs.attribs['transform'] = f"scale({new_scale_x}, {new_scale_y})"

        # Scale coordinates in path elements if any
        for element in drawing.elements:
            if isinstance(element, svgwrite.path.Path):
                path_data = element.attribs['d']
                scaled_path_data = self.scale_path_data(path_data, scale_factor)
                element.attribs['d'] = scaled_path_data

        # Save the resized SVG to the output file
        drawing.saveas(output_svg)'''
        '''# Define A4 size in millimeters (optional, adjust as needed)
        a4_width = 210
        a4_height = 297

        # Define desired viewBox values (adjust based on your SVG content)
        viewBox_x = 0  # Starting X-coordinate
        viewBox_y = 0  # Starting Y-coordinate
        viewBox_width = "100%"  # Use entire width of the SVG content 
        viewBox_height = "100%"  # Use entire height of the SVG content

        # Define preserveAspectRatio (adjust as needed)
        preserve_aspect_ratio = "xMidYMid meet"  # Example value (others available)

        # Combine viewBox and preserveAspectRatio into a single string
        viewBox_string = f"0 0 {viewBox_width} {viewBox_height}"

        with open(input_svg, 'r') as f:
            svg_content = f.read()

        # Modify the SVG content to include viewBox and preserveAspectRatio
        modified_svg_content = svg_content.replace('width="500" height="300"', f'viewBox="{viewBox_string}" preserveAspectRatio="{preserve_aspect_ratio}"')
        
        drawing.saveas(output_svg)'''

    '''def scale_path_data(self, path_data, scale_factor):
        scaled_path_data = []
        commands = path_data.split(' ')
        for command in commands:
            if command.isalpha():
                # Append command unchanged
                scaled_path_data.append(command)
            else:
                # Scale numerical values
                scaled_value = str(float(command) * scale_factor)
                scaled_path_data.append(scaled_value)
        return ' '.join(scaled_path_data)'''
       

