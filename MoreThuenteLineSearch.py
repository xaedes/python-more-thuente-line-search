# // https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/blob/master/src/common/optimization/include/optimization/line_search/more_thuente_line_search.hpp
# // Copyright 2020 the Autoware Foundation
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //  http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# //
# // Co-developed by Tier IV, Inc. and Apex.AI, Inc.


# // This file contains modified code from the following open source projects
# // published under the licenses listed below:
# //
# // Software License Agreement (BSD License)
# //
# //  Point Cloud Library (PCL) - www.pointclouds.org
# //  Copyright (c) 2010-2011, Willow Garage, Inc.
# //  Copyright (c) 2012-, Open Perception, Inc.
# //
# //  All rights reserved.
# //
# //  Redistribution and use in source and binary forms, with or without
# //  modification, are permitted provided that the following conditions
# //  are met:
# //
# //   * Redistributions of source code must retain the above copyright
# //     notice, this list of conditions and the following disclaimer.
# //   * Redistributions in binary form must reproduce the above
# //     copyright notice, this list of conditions and the following
# //     disclaimer in the documentation and/or other materials provided
# //     with the distribution.
# //   * Neither the name of the copyright holder(s) nor the names of its
# //     contributors may be used to endorse or promote products derived
# //     from this software without specific prior written permission.
# //
# //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# //  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# //  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# //  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# //  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# //  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# //  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# //  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# //  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# //  POSSIBILITY OF SUCH DAMAGE.

import dataclasses
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math

@dataclass
class FunctionValue:
    argument: float
    value: float
    derivative: float
      
@dataclass
class Function:
    evaluator: object
    jacobian_shape : tuple
    
    jacobian: np.ndarray = dataclasses.field(init=False)
        
    def __post_init__(self):
        self.jacobian = np.zeros(self.jacobian_shape, dtype=float)
    
    def evaluate(self, argument):
        value = self.evaluator(argument, self.jacobian)
        return value, self.jacobian
      
class OptimizationDirection(Enum):
    kMinimization = 1
    kMaximization = 2
    
@dataclass
class Interval:
    # an interval between unsorted points a_l and a_u.
    a_l: float
    a_u: float
    def update(self, f_t, f_l, f_u):
        # Following either "Updating Algorithm" or "Modifier Updating Algorithm" depending on the
        # provided function f (can be psi or phi).
        if f_t.value > f_l.value:
            # case a
            return Interval(f_l.argument, f_t.argument)
        elif f_t.derivative * (f_t.argument - f_l.argument) < 0:
            # case b
            return Interval(f_t.argument, f_u.argument)
        elif f_t.derivative * (f_t.argument - f_l.argument) > 0:
            # case c
            return Interval(f_t.argument, f_l.argument)
        else:
            # Converged to a point.
            return Interval(f_t.argument, f_t.argument)
          
@dataclass
class FunctionPhi:
    # This function is represented by letter phi in the paper (eq. 1.3).
    # For an underlying function f and step a_t > 0 this becomes:
    #   phi(a_t) = f(x + a_t * p)
    starting_state: np.ndarray
    initial_step: np.ndarray
    underlying_function: Function
    direction: OptimizationDirection

    multiplier: float = dataclasses.field(init=False, default=1)
    step_direction: np.ndarray = dataclasses.field(init=False)
    jacobian: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.step_direction = np.linalg.norm(self.initial_step)
        value, self.jacobian = self.underlying_function.evaluate(self.starting_state)
        derivative = self.jacobian.dot(self.step_direction)

        if self.direction == OptimizationDirection.kMinimization and derivative > 0:
            self.step_direction *= -1
        elif self.direction == OptimizationDirection.kMaximization:
            if derivative < 0:
                self.step_direction *= -1
            # The function phi must have a derivative < 0 following the introduction of the
            # More-Thuente paper. In case we want to solve a maximization problem, the derivative will
            # be positive and we need to make a dual problem from it by flipping the values of phi.
            self.multiplier = -1

    def __call__(self, step_size: float):
        # Get the value of phi for a given step.
        if step_size < 0:
            raise ValueError("Step cannot bet negative")
        current_state = self.starting_state + step_size * self.step_direction
        value, self.jacobian = self.underlying_function.evaluate(current_state)
        derivative = self.jacobian.dot(self.step_direction)
        return FunctionValue(
            step_size, 
            self.multiplier * value, 
            self.multiplier * derivative
        )

@dataclass
class FunctionPsi:
    # This class describes an auxiliary function denoted as psi in the paper (just before eq. 2.1)
    # For an objective function phi, constant mu in (0, 1), and step a_t > 0 this becomes:
    #   psi(a_t) = phi(a_t) - mu * phi(a_t).derivative * a_t

    objective_function: FunctionPhi
    mu: np.ndarray

    initial_objective_value: FunctionValue = dataclasses.field(init=False)

    def __post_init__(self):
        self.initial_objective_value = self.objective_function(0)

    def __call__(self, step_size: float):
        objective_value = self.objective_function(step_size)
        value = (
            objective_value.value
            - self.initial_objective_value.value
            - self.mu * step_size * objective_value.derivative
        )
        derivative = (
            objective_value.derivative 
            - self.mu * self.initial_objective_value.derivative
        )
        return FunctionValue(step_size, value, derivative)
      
class MoreThuenteLineSearch:
    kDelta = 0.66 # This value is used in More-Thuente paper without explanation (in the paper: Section 4, Case 3).
        
    def __init__(
            self,
            max_step,
            min_step,
            optimization_direction,
            mu = 1e-4,          # Constant \f$\mu\f$ (eq. 1.1), that forces a sufficient decrease of the function.
            eta = 0.1,          # Constant \f$\eta\f$ (eq. 1.2), that forces the curvature condition.
            max_iterations = 10 #  Default value suggested in Section 5 of the paper.
        ):
        if min_step < 0: raise ValueError("Min step cannot be negative.")
        if max_step < min_step: raise ValueError("Max step cannot be smaller than min step.")
        if mu < 0 or mu > 1: raise ValueError("mu must be in (0, 1).")
        if eta < 0 or eta > 1: raise ValueError("eta must be in (0, 1).")
        self.max_step = max_step
        self.min_step = min_step
        self.optimization_direction = optimization_direction
        self.mu = mu
        self.eta = eta
        self.max_iterations = max_iterations
        
    def compute_next_step(
            self, 
            x0,                   # Starting argument.
            initial_step,         # Initial step to initiate the search.
            optimization_problem  # The optimization problem for generating values of function denoted as f in the paper.
        ):
        # Returns: The new step to make in order to optimize the function.
        # If the length of initial_step is smaller than the m_step_min then an unmodified
        # initial_step will be returned.
        
        a_t = min(np.linalg.norm(initial_step), self.max_step)
        if a_t < self.min_step:
            # We don't want to perform the line search as the initial step is out of allowed bounds. We
            # assume that the optimizer knows what it is doing and return the initial_step unmodified.
            return initial_step
        
        interval = Interval(self.min_step, self.max_step)
        
        phi = FunctionPhi(x0, initial_step, optimization_problem, self.optimization_direction)
        psi = FunctionPsi(phi, self.mu)
        
        phi_0 = phi(0)
        phi_t = phi(a_t)
        psi_t = psi(a_t)
        f_l = psi(interval.a_l)
        f_u = psi(interval.a_u)
        
        use_auxiliary = True
        
        # Follows the "Search Algorithm" as presented in the paper.
        for step_iterations in range(self.max_iterations):
            if psi_t.value <= 0 and abs(phi_t.derivative) <= self.eta * abs(phi_0.derivative):
                # We reached the termination condition as the step satisfies the strong Wolfe conditions (the
                # ones in the if condition). This means we have converged and are ready to return the found
                # step.
                break
            
            # Pick next step size by interpolating either phi or psi depending on which update algorithm is
            # currently being used.
            if use_auxiliary:
                a_t = self.find_next_step_length(psi_t, f_l, f_u)
            else:
                a_t = self.find_next_step_length(phi_t, f_l, f_u)
                
            if a_t < self.min_step or math.isnan(a_t):
                # This can happen if we are closer than the minimum step to the optimum. We don't want to do
                # anything in this case.
                a_t = 0
                break
                
            phi_t = phi(a_t)
            psi_t = psi(a_t)
    
            # Decide if we want to switch to using a "Modified Updating Algorithm" (shown after theorem 3.2
            # in the paper) by switching from using function psi to using function phi. The decision
            # follows the logic in the paragraph right before theorem 3.3 in the paper.
            if use_auxiliary and (psi_t.value <= 0 and psi_t.derivative > 0):
                use_auxiliary = False
                # We now want to switch to using phi so compute the required values.
                f_l = phi(interval.a_l)
                f_u = phi(interval.a_u)
                
            if use_auxiliary:
                # Update the interval that will be used to generate the next step using the
                # "Updating Algorithm" (right after theorem 2.1 in the paper).
                interval = interval.update(psi_t, f_l, f_u)
                f_l = psi(interval.a_l)
                f_u = psi(interval.a_u)
            else:
                # Update the interval that will be used to generate the next step using the
                # "Modified Updating Algorithm" (right after theorem 3.2 in the paper).
                interval = interval.update(psi_t, f_l, f_u)
                f_l = phi(interval.a_l)
                f_u = phi(interval.a_u)
            if self.approx_eq(interval.a_u, interval.a_l, self.min_step):
                # The interval has converged to a point so we can stop here.
                a_t = interval.a_u
                break
        return a_t * phi.step_direction
    

    def find_next_step_length(self, f_t, f_l, f_u):
        if any([math.isnan(f_t.argument), math.isnan(f_l.argument), math.isnan(f_u.argument)]):
            raise ValueError("Got nan values in the step computation function.")
        
        kValueEps = 0.00001
        kStepEps  = 0.00001
        
        # calculate the minimizer of the cubic that interpolates f_a, f_a_derivative, f_b and
        # f_b_derivative on [a, b]. Equation 2.4.52 [Sun, Yuan 2006]
        def find_cubic_minimizer(f_a, f_b):
            if self.approx_eq(f_a.argument, f_b.argument, kStepEps):
                return f_a.argument
            z = 3 * (f_a.value - f_b.value) / (f_b.argument - f_a.argument) + f_a.derivative + f_b.derivative
            w = math.sqrt(z*z - f_a.derivative * f_b.derivative)
            # Equation 2.4.56 [Sun, Yuan 2006]
            return (
                f_b.argument 
                - (f_b.argument - f_a.argument)
                * (f_b.derivative + w - z)
                / (f_b.derivative - f_a.derivative + 2*w)
            )
        
        # calculate the minimizer of the quadratic that interpolates f_a, f_b and f'_a
        def find_a_q(f_a, f_b):
            if self.approx_eq(f_a.argument, f_b.argument, kStepEps):
                return f_a.argument
            
            return (
                f_a.argument 
                + 0.5 
                * (f_b.argument - f_a.argument) 
                * (f_b.argument - f_a.argument)
                * f_a.derivative
                / (f_a.value - f_b.value + (f_b.argument - f_a.argument) * f_a.derivative)
            )
        
        # calculate the minimizer of the quadratic that interpolates f'_a, and f'_b
        def find_a_s(f_a, f_b):
            if self.approx_eq(f_a.argument, f_b.argument, kStepEps):
                return f_a.argument
                             
            return (
                f_a.argument
                + (f_b.argument - f_a.argument)
                * f_a.derivative
                / (f_a.derivative - f_b.derivative)
            )
        
        # We cover here all the cases presented in the More-Thuente paper in section 4.
        if f_t.value > f_l.value: 
            # Case 1 from section 4.
            a_c = find_cubic_minimizer(f_l, f_t)
            a_q = find_a_q(f_l, f_t)
            if abs(a_c - f_l.argument) < abs(a_q - f_l.argument):
                return a_c
            else:
                return 0.5 * (a_q + a_c)
        elif f_t.derivative * f_l.derivative < 0: 
            # Case 2 from section 4.
            a_c = find_cubic_minimizer(f_l, f_t)
            a_s = find_a_s(f_l, f_t)
            if abs(a_c - f_t.argument) >= abs(a_s - f_t.argument):
                return a_c
            else:
                return a_s
        elif self.approx_lte(abs(f_t.derivative), abs(f_l.derivative), kValueEps):
            # Case 3 from section 4.
            a_c = find_cubic_minimizer(f_l, f_t)
            a_s = find_a_s(f_l, f_t)
            if abs(a_c - f_t.argument) < abs(a_s - f_t.argument):
                return min(
                    f_t.argument + self.kDelta * (f_u.argument - f_t.argument),
                    a_c
                )
            else:
                return max(
                    f_t.argument + self.kDelta * (f_u.argument - f_t.argument),
                    a_s
                )
        else:
            # Case 4 from section 4.
            return find_cubic_minimizer(f_t, f_u)
        
    def approx_eq(self, a, b, eps):
        return abs(a-b) < eps
    
    def approx_lte(self, a, b, eps):
        return a <= b + eps
      
