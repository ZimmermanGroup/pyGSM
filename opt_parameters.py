import options
from _linesearch import backtrack

class parameters:
    """the parameters of optimizers
    	epsilon: epsilon for the convergence test
    	ftol: A parameter to control the accuracy of the line search routine.The default value is 1e-4. 
    		This parameter should be greaterthan zero and smaller than 0.5.
    	wolfe: A coefficient for the Wolfe condition.The default value is 0.9. 
    		This parameter should be greater the ftol parameter and smaller than 1.0.
    	min_step: The minimum step of the line search routine.
    	max_step: The maximum step of the line search routine."""

    @staticmethod
    def default_options():
        """ ef default options. """

        if hasattr(parameters, '_default_options'): return parameters._default_options.copy()
        opt = options.Options() 


        opt.add_option(
                key='opt_type',
                required=True,
                value="UNCONSTRAINED",
                allowed_types=[str],
                allowed_values=["UNCONSTRAINED", "ICTAN", "CLIMB", "TS", "MECI", "SEAM", "TS-SEAM"],
                doc='The type of unconstrained optimization'
                )

        opt.add_option(
                key='OPTTHRESH',
                value=0.0005,
                required=False,
                allowed_types=[float],
                doc='Convergence threshold'
                )

        opt.add_option(
                key='DMAX',
                value=0.1
                )

        opt.add_option(
                key='SCALEQN',
                value=1,
                )

        opt.add_option(
                key='Linesearch',
                value=backtrack
                )

        opt.add_option(
                key='MAXAD',
                value=0.075,
                )

        parameters._default_options = opt
        return parameters._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return parameters(parameters.default_options().set_values(kwargs))

    def __init__(self,
            options,
            ):

        self.options = options
        
        self.epsilon=1e-5
        self.ftol=1e-4
        self.wolfe=0.9 
        self.max_linesearch=10
        self.min_step=1e-20
        self.max_step=1e20

        # additional parameters needed by linesearch
        self.conv_disp = 12e-4 #max atomic displacement
        self.conv_gmax = 3e-4 #max gradient
        self.conv_Ediff = 1e-6 #E diff
        self.conv_grms = options['OPTTHRESH']

