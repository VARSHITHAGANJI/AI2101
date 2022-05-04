

import numpy as np 
import cvxpy as cp

x = cp.Variable((2,1))
P = np.array([[2,-1],[-1,4]])
q = np.array([[-1,0]]).T
A = np.array([[1,-2],[1,4],[5,-76]])
b = np.array([[-2],[-3],[1]])
obj = 1/2*cp.quad_form(x,P) + q.T@x
fn = cp.Minimize(obj)
c = [A@x <= b]
pr = cp.Problem(cp.Minimize(obj),c)
pr.solve()

print('optimalx value:',x.value)




s = np.linalg.inv(P)


s = np.linalg.inv(P)

lam = cp.Variable((3,1))
z = s@(A.T@lam+q)

dual  = -((lam.T)@b) - ((lam.T)@(A@z)) - (q.T@s) + 1/2*(z.T@P@z)

f = cp.Maximize(dual[0,0])
c = [lam >= 0]
p = cp.Problem(f,c)
p.solve()

print('dual optimal variable value: ',lam.value[0])

