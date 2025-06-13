import numpy as np
from scipy import signal
import scipy.linalg as la

A_c = np.array([[0, 1, 0, 0],   # ẋ  = v_x
            [0, 0, 0, 0],   # v̇_x = a
            [0, 0, 0, 1],   # ẏ  = v_y
            [0, 0, 0, 0]])  # v̇_y = δ

B_c = np.array([[0, 0],         # a   wirkt nur auf v̇_x
            [1, 0],
            [0, 0],         # δ   wirkt nur auf v̇_y
            [0, 1]])

class LQR:

    def __init__(self, Ts = 0.5, Q = np.diag([ 20, 10, 10]) , R = np.diag([5, 10])):

        A, B, *_ = signal.cont2discrete(
                (A_c, B_c, np.zeros((4, 2)), np.zeros((2, 2))),
                Ts, method="zoh"
                    )
        
        self.A = np.array([   # ẋ  = v_x
            [1, 0, 0],   # v̇_x = a
            [0, 1, Ts],
               [0, 0, 1]   # ẏ  = v_y
          ])  # v̇_y = δ

        self.B = np.array([      # a   wirkt nur auf v̇_x
                [Ts, 0],
                [0, 0.5*Ts**2], 
                  [0, Ts]  ])



        P = la.solve_discrete_are(self.A, self.B, Q, R)
        self.K = np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        self.z_goal = np.array([  13 ,0, 0], dtype = float)

    

    def updateGoal(self, deltaZ):
        self.z_goal = deltaZ
        
    def updateState(self, z):
        u = -self.K @ (z - self.z_goal)

        u[0]     = max(-5, min( 3, round(u[0], 2)))
        u[1]     = max(-0.3, min(0.3, u[1]))

        z_next = self.A @ z + self.B @ u
        
        vx_new = max(0, min( 14, z_next[0]))
        z_next[0] = vx_new
        return np.round(z_next, 2)  , u

       
