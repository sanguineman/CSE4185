import numpy as np

epsilon = 1e-3

def OOB(x, y, M, N): # check if the coordinates out of bound, returns True if it is, False otherwise.
    if x < 0 or x >= M or y < 0 or y >= N:
        return True
    else:
        return False

dx = [0,-1,0,1] # left, up, right, down
dy = [-1,0,1,0] # left, up, right, down

def compute_transition_matrix(model):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """
    # raise RuntimeError("You need to write this part!")
    M, N = model.M, model.N
    P = np.zeros([M, N, 4, M, N])

    for i in range(M):
        for j in range(N):
            if model.T[i, j] == True: # if the state is the terminal state, make the transition probability zero
                P[i,j,:,:,:] = 0
                continue
            for a in range(4): 
                # intended direction
                nx = i + dx[a]
                ny = j + dy[a]
                if OOB(nx, ny, M, N) or model.W[nx, ny]: # if next state is OOB or wall
                    P[i,j,a,i,j] += model.D[i,j,0] # probability is added to the same state
                else:
                    P[i,j,a,nx,ny] += model.D[i,j,0] # probability is added to the next state
                # unintended counter-clockwise
                nx = i + dx[(a+3)%4]
                ny = j + dy[(a+3)%4]
                if OOB(nx, ny, M, N) or model.W[nx, ny]: # if next state is OOB or wall
                    P[i,j,a,i,j] += model.D[i,j,1] # # probability is added to the same state
                else:
                    P[i,j,a,nx,ny] += model.D[i,j,1] # probability is added to the next state
                # unintended clockwise
                nx = i + dx[(a+1)%4]
                ny = j + dy[(a+1)%4]
                if OOB(nx, ny, M, N) or model.W[nx, ny]: # if next state is OOB or wall
                    P[i,j,a,i,j] += model.D[i,j,2] # probability is added to the same state
                else:
                    P[i,j,a,nx,ny] += model.D[i,j,2] # probability is added to the next state

    return P


def update_utility(model, P, U_current):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    """
    U_next = np.zeros_like(U_current) # U_next same shape as U_current
    # raise RuntimeError("You need to write this part!")
    M, N = model.M, model.N
    for i in range(M):
        for j in range(N):
            Max = -10000000 # initialize with very small value
            for a in range(4): # to pull out the max out of all actions
                # intended direction
                Sum = 0
                # stay at the same spot
                Sum += (P[i,j,a,i,j] * U_current[i,j])

                # check all four directions
                for s_p in range(4):
                    nx = i + dx[s_p]
                    ny = j + dy[s_p]
                    if OOB(nx,ny, M, N) or model.W[nx, ny]: # if the next state is OOB
                        continue
                    Sum += (P[i,j,a,nx,ny] * U_current[nx,ny]) # even if the direction is 180 degree opposite of intended action, transition probability is 0, resulting in 0 addition, so it is fine.
                
                if Sum > Max: # renew the max value
                    Max = Sum
            U_next[i, j] = model.R[i,j] + model.gamma * Max # update the utility of that state

    return U_next


def value_iteration(model):
    """
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    """
    P = compute_transition_matrix(model) # calculate the transition probabiltiy
    M, N = model.M, model.N
    U_current = np.zeros([M,N])
    U_next = np.zeros_like(U_current) # U_next same shape as U_current
    iter_num = 0 # track the number of iteration
    converged = False # check if it is converged
    while iter_num < 100 and converged == False: # repeat until the iteration num is smaller than 100 and it is not converged yet 
        U_next = update_utility(model, P, U_current) # update the utility
        check = True
        for i in range(M):
            for j in range(N):
                if abs(U_next[i,j] - U_current[i,j]) >= epsilon: # if one of the state's difference in utility is greater than or equal to epsilon, break instantly
                    check = False
                    break
            if check == False:
                break
        
        if check == True: # if any of the state's difference in utility is greater than or equal to epsilon, it means it is converged, so terminate the loop
            converged = True
        
        U_current = U_next
        iter_num += 1
    # raise RuntimeError("You need to write this part!")
    return U_next # return the result of the value iteration
