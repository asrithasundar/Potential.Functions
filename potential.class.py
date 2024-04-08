import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from numpy import gradient

class Potential:
    def __init__(self,*args):
        self.coefficients=args

    def f(self,x: ArrayLike):
        #to calculate the potential function,which will be overridden by subclasses
        raise NotImplementedError("subclass must  be implemented abstract method")

    def analytical_derivative(self, x: ArrayLike):
        raise NotImplementedError("subclass must  be implemented abstract method")


    def numerical_derivative(self, x: ArrayLike, h=0.0001):
        return (self.f(x+h)-self.f(x-h))/(2*h)


    def plot_function(self,x_values)-> None:
         """
         plot the potential function over a given range of x values
     """


         y_values = self.f(x_values)
         dy_analytical = self.analytical_derivative(x_values)
         dy_numerical = self.numerical_derivative(x_values)

         plt.plot(x_values, y_values, label="f(x)", color="blue", linewidth=2, marker=".", markerfacecolor="k",
                  markersize=4)

         plt.plot(x_values, dy_analytical, label="f'(x) - analytical", color="red", linewidth=2, marker=".", markerfacecolor="k",
                  markersize=4)
         plt.plot(x_values, dy_numerical, label="f'(x) - numerical", color="green", linewidth=2, marker=".",
                  markerfacecolor="k",
                  markersize=4)


         plt.title(f"{self.__class__.__name__}  and Derivatives Plot")

         plt.xlabel("x")
         plt.ylabel("f(x)/f'(x)")
         plt.savefig(f"{self.__class__.__name__} and Derivatives fig.pdf")
         plt.legend()
         plt.show()

class Linear_Potential(Potential):
    def f(self, x: ArrayLike):
        m, c = self.coefficients
        return m * x + c


    def analytical_derivative(self,x):
        m ,_ = self.coefficients
        return  np.full_like(x, m)


class Quadratic_Potential(Potential):
    def f(self, x: ArrayLike):
        a, b, c = self.coefficients
        return a * x**2 + b * x + c

    def analytical_derivative(self,x):
        a,b,_= self.coefficients
        return a * 2 * x + b
class DoubleWell_Potential(Potential):
    def f(self, x: ArrayLike):
        a, b, c = self.coefficients
        return a * x**4 - b * x**2 + c

    def analytical_derivative(self,x):
        a,b,_= self.coefficients
        return a * 4 * x ** 3 - b * 2 * x

linear = Linear_Potential(2, -3)
linear.plot_function(np.linspace(-10,10,100))
quadratic = Quadratic_Potential(1, -5, 6)  # Quadratic potential with a=1, b=-5, c=6
quadratic.plot_function(np.linspace(-10,15,100))
doublewell = DoubleWell_Potential(1,2,-1)
doublewell.plot_function(np.linspace(-1.5,1.5,100))