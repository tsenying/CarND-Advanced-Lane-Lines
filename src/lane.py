from line import Line
import config

class Lane():
    def __init__(self, nframes):
        # number of frames in history
        self.nframes = nframes
        
        # left and right lines
        self.left_line = Line( self.nframes, 'left' )
        self.right_line = Line( self.nframes, 'right' )
        
        # radius of curvature of the lane (meters) (combined left and right line curvatures)
        self.radius_of_curvature = None 
        
        # distance of vehicle center from lane center (meters)
        self.center_offset = None
        
    def update( self ):
        left_valid = self.left_line.update( self.right_line )
        right_valid = self.right_line.update( self.left_line )
        
        is_valid = left_valid and right_valid
        if not is_valid:
            config.debug_log.write("Lane:update left_valid={}, right_valid={}\n".format(left_valid, right_valid))
        return is_valid
        
        