import numpy as np
import config
from lane_fit_utils import are_lines_parallel, line_distance

# Define a class to receive the characteristics of each line detection
class Line():
    y_eval = config.image_shape['height'] - 1;
    
    def __init__(self, nframes, name):
        self.name = name
        
        # number of frames in history
        self.nframes = nframes
        
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = None  
        
        #y values for detected line pixels
        self.ally = None
        
        # Polynomial function for current_fit coefficients
        self.current_fit_poly = None
        # Polynomial function for best_fit coefficients
        self.best_fit_poly = None
    
    def update( self, other = None ):
        """
        Returns:
            boolean: False if bad line
        """
        is_valid = True
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            if self.valid( other ):
                self.best_fit = ( self.best_fit * (self.nframes - 1) + self.current_fit) / self.nframes
            else:
                is_valid = False
        #print("Line#update best_fit={}, current_fit={}".format(self.best_fit, self.current_fit))
        
        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)
        
        return is_valid
        
    def valid( self, other=None):
        """
        check current fit 
        - is line curvature similar to best_fit?
        - is line a reasonable distance from other lane line?

        Args:

        Returns:
            boolean
        """
        
        # see if current fit is similar to history
        # threshold=(0.0003, 0.55)
        is_parallel = False
        #if (are_lines_parallel( self.current_fit, self.best_fit, threshold=(0.0005, 0.55) )):
        #if (are_lines_parallel( self.current_fit, self.best_fit, threshold=(0.00057, 0.77) )):
        if (are_lines_parallel( self.current_fit, self.best_fit, threshold=(0.00065, 0.8) )):
            is_parallel = True
        else:
            config.debug_log.write("Line({})#valid Frame {} invalid, is_parallel={}, current_fit={}, best_fit={}, fit_diff={}\n".format( self.name, config.count, is_parallel, self.current_fit, self.best_fit, (self.current_fit - self.best_fit) ))
            
        distance_from_other_line_ok = False
        other_line_distance = line_distance(self.current_fit, other.best_fit, self.y_eval)

        lane_width_range = (600, 720)
        if lane_width_range[0] <= other_line_distance <= lane_width_range[1]:
            distance_from_other_line_ok = True
        else:
            config.debug_log.write("Line({})#valid Frame {} invalid, other_line_distance {}\n".format( self.name, config.count, other_line_distance ))
            
        is_valid = is_parallel and distance_from_other_line_ok
        if not is_valid:
            config.debug_log.write("Line({})#valid is_parallel={}, distance_from_other_line_ok={}\n".format( self.name, is_parallel, distance_from_other_line_ok))
        
        return is_valid
            

    def is_current_fit_parallel(self, other_line, threshold=(0.0005, 0.55)):
        """
        check if two lines are parallel by comparing first two polynomial fit coefficients
        
        Args:
            param other_line (array): line to compare polynomial coefficients
            threshold (tuple): floats representing delta thresholds for coefficients
        
        Returns:
            boolean
        """
        diff_first_coefficient_ = np.abs(self.current_fit[0] - other_line.current_fit[0])
        diff_second_coefficient = np.abs(self.current_fit[1] - other_line.current_fit[1])

        is_parallel = diff_first_coefficient < threshold[0] and diff_second_coefficient < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):
        """
        get distance between current fit with other_line
        
        Args:
            other_line:
        Returns:
            float
        """
        return np.abs(self.current_fit_poly(y_eval) - other_line.current_fit_poly(y_eval))

    def get_best_fit_distance(self, other_line):
        """
        get the distance between best fit with other line
        
        Args:
            other_line:
        Returns:
            float
        """
        return np.abs(self.best_fit_poly(y_eval) - other_line.best_fit_poly(y_eval))
    
