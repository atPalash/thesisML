import math

import numpy as np

x = [278, 273, 275, 272, 272, 271, 270, 269, 267, 266, 264, 264, 263, 263, 260]
y = [386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400]

z = np.polyfit(x, y, 1)
print math.degrees(math.atan(z[0]))
print z