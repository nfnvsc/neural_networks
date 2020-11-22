class Function:
    def __init__(self):
        self._function = None
        self._derivative = None
        
    def calculate(self, x):
        return self._function(x)
    def derivative(self, x):
        return self._derivative(x)