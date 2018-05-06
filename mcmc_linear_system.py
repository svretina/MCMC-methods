#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt



A_tilt = np.array([[0.2,0.1],[0.1,-0.3]])
b = np.array([0.5,1.1])

p = np.absolute(A_tilt)
v = A_tilt/np.absolute(A_tilt)

last_collumn = 1-np.sum(p,axis=1)
last_collumn = np.reshape(last_collumn,(len(last_collumn),1))

p = np.append(p,last_collumn,axis=1)

last_row = np.zeros(len(p[1,:]))
last_row[-1] = 1
last_row = np.reshape(last_row,(1,len(last_row)))

p = np.append(p,last_row,axis=0)

# print("P_ij matrix=\n",p)
# print("v_ij matrix=\n",v)


cumulated_propabilities = np.cumsum(p,axis=1)
#print(cumulated_propabilities)

number_of_solutions = A_tilt.shape[0]
number_of_random_paths = 1000 #user defined
beta = np.zeros(1)
v_xi = np.zeros(1)
PTi = np.zeros(1)
solution = np.zeros((number_of_solutions,1))
var_sol = np.zeros((number_of_solutions,1))
for i in range(number_of_solutions):
    xi = np.zeros((number_of_random_paths,1))
    for j in range(number_of_random_paths):
        current_state = i
        # initializing b,v arrays 
        beta[0] = b[current_state]
        v_xi[0] = 1
        while(current_state<(A_tilt.shape[0])):
            random_number = np.random.random()
            # check random number and cumulated propability matrix to jump to the right State 
            for k in range(cumulated_propabilities.shape[1]-1):
                if random_number< cumulated_propabilities[current_state,0]:
                    previous_state = current_state
                    current_state = current_state
                    break
                elif (cumulated_propabilities[current_state,k]< random_number) and (random_number<cumulated_propabilities[current_state,k+1]):
                    previous_state = current_state
                    current_state = k+1
                    break
            # create the b and v arrays for calculatin xi
            if current_state < A_tilt.shape[0]:
                v_xi = np.append(v_xi,v[previous_state,current_state])
                beta = np.append(beta,b[current_state])
        v_xi = np.cumprod(v_xi)
        xi[j,0] = np.dot(beta,v_xi)
        # clearing arrays b,v
        beta = np.delete(beta,np.s_[1:])
        v_xi = np.delete(v_xi,np.s_[1:])
    # calculating solution
    solution[i,0] = np.mean(xi)
    var_sol[i,0] = np.var(xi,ddof=1)
print("Solution=\n",solution[0][0],"+/-",var_sol[0][0])
print(solution[1][0],"+/-",var_sol[1][0])
