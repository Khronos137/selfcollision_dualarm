import numpy as np

def derCinemDirecta6ManH(L, q):
    h, b, l1, l1b, l2, l3 = L[0], L[1], L[2], L[3], L[4], L[5]
    l1a = (l1**2 + l1b**2)**0.5
    
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2]+0.7854, q[3], q[4], q[5]  # Ajuste de q3 para compensar el ángulo de 45 grados (0.7854 radianes) 

    hx = (l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1)) 
          - l3*(np.sin(q5)*(np.sin(q1)*np.sin(q4) - np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1)
          - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))) - np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1)
          + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))) + l1a*np.sin(q2 + 0.113)*np.cos(q1) 
          - l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) + l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))

    hy = (- b - l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)) 
          - np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67))) 
          - l2*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)) 
          - l1a*np.cos(q2 + 0.113) - l1b*np.cos(q3 + 0.67)*np.sin(q2 + 0.113) - l1b*np.cos(q2 + 0.113)*np.sin(q3 + 0.67))

    hz = (h + l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1)) 
          + l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1))
          + np.sin(q5)*(np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) 
          - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) + np.cos(q1)*np.sin(q4))) + l1a*np.sin(q2 + 0.113)*np.sin(q1) 
          - l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) + l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1))

    return np.array([hx, hy, hz])


def derJac6ManH(L, q):
    h, b, l1, l1b, l2, l3 = L[0], L[1], L[2], L[3], L[4], L[5]
    l1a = (l1**2 + l1b**2)**0.5
    
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2]+0.7854, q[3], q[4], q[5]  # Ajuste de q3 para compensar el ángulo de 45 grados (0.7854 radianes)

    J11 = (l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1)) + np.sin(q5)*(np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) + np.cos(q1)*np.sin(q4))) - l1a*np.sin(q2 + 0.113)*np.sin(q1) - l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1)) - l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1))
    J12 = (l2*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1)) + l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1)) - np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))) + l1a*np.cos(q2 + 0.113)*np.cos(q1) + l1b*np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + l1b*np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))
    J13 = (l2*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1)) + l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1)) - np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))) + l1b*np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + l1b*np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))
    J14 = (-l3*np.sin(q5)*(np.cos(q4)*np.sin(q1) + np.sin(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))))
    J15 = (-l3*(np.cos(q5)*(np.sin(q1)*np.sin(q4) - np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))) + np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))))
    J16 = 0.0

    J21 = 0.0
    J22 = (l1a*np.sin(q2 + 0.113) + l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)) + np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113))) + l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)) - l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113) + l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113))
    J23 = (l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)) + np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113))) + l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)) - l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113) + l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113))
    J24 = (-l3*np.sin(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)))
    J25 = (l3*(np.sin(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)) + np.cos(q4)*np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67))))
    J26 = 0.0

    J31 = (l2*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1)) - l3*(np.sin(q5)*(np.sin(q1)*np.sin(q4) - np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))) - np.cos(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.cos(q1))) + l1a*np.sin(q2 + 0.113)*np.cos(q1) - l1b*np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.cos(q1) + l1b*np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.cos(q1))
    J32 = (l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) - np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1))) + l2*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) + l1a*np.cos(q2 + 0.113)*np.sin(q1) + l1b*np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + l1b*np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1))
    J33 = (l3*(np.cos(q5)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) - np.cos(q4)*np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1))) + l2*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) + l1b*np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + l1b*np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1))
    J34 = (l3*np.sin(q5)*(np.cos(q1)*np.cos(q4) - np.sin(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1))))
    J35 = (-l3*(np.sin(q5)*(np.cos(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1) + np.cos(q2 + 0.113)*np.sin(q3 + 0.67)*np.sin(q1)) - np.cos(q5)*(np.cos(q4)*(np.cos(q3 + 0.67)*np.cos(q2 + 0.113)*np.sin(q1) - np.sin(q3 + 0.67)*np.sin(q2 + 0.113)*np.sin(q1)) + np.cos(q1)*np.sin(q4))))
    J36 = 0.0

    Jr = np.array([
        [J11, J12, J13, J14, J15, J16],
        [J21, J22, J23, J24, J25, J26],
        [J31, J32, J33, J34, J35, J36]
    ])

    return Jr