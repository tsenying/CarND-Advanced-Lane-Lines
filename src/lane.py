from line import Line

class Lane():
    def __init__(self, nframes):
        # number of frames in history
        self.nframes = nframes
        
        # left and right lines
        self.left_line = Line( self.nframes )
        self.right_line = Line( self.nframes )
        
        # radius of curvature of the lane (meters) (combined left and right line curvatures)
        self.radius_of_curvature = None 
        
        # distance of vehicle center from lane center (meters)
        self.center_offset = None
        
    def update( self ):
        self.left_line.update()
        self.right_line.update()
        
        