from line import Line

class Lane():
    def __init__(self, nframes):
        # number of frames in history
        self.n = nframes
        
        # left and right lines
        self.left_line = Line()
        self.right_line = Line()
        
        # radius of curvature of the lane (meters) (combined left and right line curvatures)
        self.radius_of_curvature = None 
        
        # distance of vehicle center from lane center (meters)
        self.center_offset = None
        
        