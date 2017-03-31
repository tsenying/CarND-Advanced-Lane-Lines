count = 0
lane_finder = None

image_shape = { 'width': 1280, 'height': 720 }

# Define conversions in x and y from pixels space to meters
REAL2PIXELS = {
  'ym_per_pix': 30/720, # meters per pixel in y dimension
  'xm_per_pix': 3.7/700 # meters per pixel in x dimension
}

log_file = None
