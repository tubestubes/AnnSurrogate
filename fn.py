"""
Defines the function to be analysed
can be imported to other scripts
"""
def eqn(a,b,c,d,e,f,g):
     f = 15.59*(10**4) - ( ( a*(b**2) )/( 2*(b**3) ) )*( ( d**2 - 4*e*f*(g**2) + d*(f + 4*e + 2*f*g) )/( d*e*(d+f+2*f*g) ))
     return f

# This can test if it runs, compare with manual calc
if __name__=='__main__':
     print(eqn(1,1,1,1,1,1,1))