import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt
import sympy as smp
from sympy import symbols
import math
    
y = symbols("y")
x = symbols("x")
alpha = 0.1
a = 0
b = 1
yL = 1
yR =1 

# refx = np.array([0.        , 0.00390625, 0.0078125 , 0.01171875, 0.015625  ,
#        0.01953125, 0.0234375 , 0.02734375, 0.03125   , 0.03515625,
#        0.0390625 , 0.04296875, 0.046875  , 0.05078125, 0.0546875 ,
#        0.05859375, 0.0625    , 0.06640625, 0.0703125 , 0.07421875,
#        0.078125  , 0.08203125, 0.0859375 , 0.08984375, 0.09375   ,
#        0.09765625, 0.1015625 , 0.10546875, 0.109375  , 0.11328125,
#        0.1171875 , 0.12109375, 0.125     , 0.12890625, 0.1328125 ,
#        0.13671875, 0.140625  , 0.14453125, 0.1484375 , 0.15234375,
#        0.15625   , 0.16015625, 0.1640625 , 0.16796875, 0.171875  ,
#        0.17578125, 0.1796875 , 0.18359375, 0.1875    , 0.19140625,
#        0.1953125 , 0.19921875, 0.203125  , 0.20703125, 0.2109375 ,
#        0.21484375, 0.21875   , 0.22265625, 0.2265625 , 0.23046875,
#        0.234375  , 0.23828125, 0.2421875 , 0.24609375, 0.25      ,
#        0.25390625, 0.2578125 , 0.26171875, 0.265625  , 0.26953125,
#        0.2734375 , 0.27734375, 0.28125   , 0.28515625, 0.2890625 ,
#        0.29296875, 0.296875  , 0.30078125, 0.3046875 , 0.30859375,
#        0.3125    , 0.31640625, 0.3203125 , 0.32421875, 0.328125  ,
#        0.33203125, 0.3359375 , 0.33984375, 0.34375   , 0.34765625,
#        0.3515625 , 0.35546875, 0.359375  , 0.36328125, 0.3671875 ,
#        0.37109375, 0.375     , 0.37890625, 0.3828125 , 0.38671875,
#        0.390625  , 0.39453125, 0.3984375 , 0.40234375, 0.40625   ,
#        0.41015625, 0.4140625 , 0.41796875, 0.421875  , 0.42578125,
#        0.4296875 , 0.43359375, 0.4375    , 0.44140625, 0.4453125 ,
#        0.44921875, 0.453125  , 0.45703125, 0.4609375 , 0.46484375,
#        0.46875   , 0.47265625, 0.4765625 , 0.48046875, 0.484375  ,
#        0.48828125, 0.4921875 , 0.49609375, 0.5       , 0.50390625,
#        0.5078125 , 0.51171875, 0.515625  , 0.51953125, 0.5234375 ,
#        0.52734375, 0.53125   , 0.53515625, 0.5390625 , 0.54296875,
#        0.546875  , 0.55078125, 0.5546875 , 0.55859375, 0.5625    ,
#        0.56640625, 0.5703125 , 0.57421875, 0.578125  , 0.58203125,
#        0.5859375 , 0.58984375, 0.59375   , 0.59765625, 0.6015625 ,
#        0.60546875, 0.609375  , 0.61328125, 0.6171875 , 0.62109375,
#        0.625     , 0.62890625, 0.6328125 , 0.63671875, 0.640625  ,
#        0.64453125, 0.6484375 , 0.65234375, 0.65625   , 0.66015625,
#        0.6640625 , 0.66796875, 0.671875  , 0.67578125, 0.6796875 ,
#        0.68359375, 0.6875    , 0.69140625, 0.6953125 , 0.69921875,
#        0.703125  , 0.70703125, 0.7109375 , 0.71484375, 0.71875   ,
#        0.72265625, 0.7265625 , 0.73046875, 0.734375  , 0.73828125,
#        0.7421875 , 0.74609375, 0.75      , 0.75390625, 0.7578125 ,
#        0.76171875, 0.765625  , 0.76953125, 0.7734375 , 0.77734375,
#        0.78125   , 0.78515625, 0.7890625 , 0.79296875, 0.796875  ,
#        0.80078125, 0.8046875 , 0.80859375, 0.8125    , 0.81640625,
#        0.8203125 , 0.82421875, 0.828125  , 0.83203125, 0.8359375 ,
#        0.83984375, 0.84375   , 0.84765625, 0.8515625 , 0.85546875,
#        0.859375  , 0.86328125, 0.8671875 , 0.87109375, 0.875     ,
#        0.87890625, 0.8828125 , 0.88671875, 0.890625  , 0.89453125,
#        0.8984375 , 0.90234375, 0.90625   , 0.91015625, 0.9140625 ,
#        0.91796875, 0.921875  , 0.92578125, 0.9296875 , 0.93359375,
#        0.9375    , 0.94140625, 0.9453125 , 0.94921875, 0.953125  ,
#        0.95703125, 0.9609375 , 0.96484375, 0.96875   , 0.97265625,
#        0.9765625 , 0.98046875, 0.984375  , 0.98828125, 0.9921875 ,
#        0.99609375, 1.        ])

# ref = np.array([9.86706721e-07, 3.44954056e-06, 2.66540985e-06, 3.32943500e-06,
#        3.15406582e-06, 3.32684624e-06, 3.31313610e-06, 3.36782918e-06,
#        3.38182572e-06, 3.40675636e-06, 3.42185941e-06, 3.43750063e-06,
#        3.45005239e-06, 3.46170707e-06, 3.47197151e-06, 3.48139488e-06,
#        3.48999820e-06, 3.49797113e-06, 3.50538900e-06, 3.51234440e-06,
#        3.51889863e-06, 3.52510786e-06, 3.53101607e-06, 3.53666105e-06,
#        3.54207413e-06, 3.54728208e-06, 3.55230764e-06, 3.55717038e-06,
#        3.56188718e-06, 3.56647267e-06, 3.57093964e-06, 3.57529928e-06,
#        3.57956143e-06, 3.58373479e-06, 3.58782708e-06, 3.59184516e-06,
#        3.59579516e-06, 3.59968256e-06, 3.60351229e-06, 3.60728880e-06,
#        3.61101609e-06, 3.61469779e-06, 3.61833720e-06, 3.62193733e-06,
#        3.62550090e-06, 3.62903041e-06, 3.63252815e-06, 3.63599622e-06,
#        3.63943655e-06, 3.64285091e-06, 3.64624095e-06, 3.64960816e-06,
#        3.65295395e-06, 3.65627962e-06, 3.65958637e-06, 3.66287531e-06,
#        3.66614749e-06, 3.66940388e-06, 3.67264537e-06, 3.67587281e-06,
#        3.67908699e-06, 3.68228865e-06, 3.68547847e-06, 3.68865711e-06,
#        3.69182518e-06, 3.69498324e-06, 3.69813184e-06, 3.70127149e-06,
#        3.70440266e-06, 3.70752580e-06, 3.71064134e-06, 3.71374968e-06,
#        3.71685120e-06, 3.71994626e-06, 3.72303521e-06, 3.72611836e-06,
#        3.72919602e-06, 3.73226849e-06, 3.73533603e-06, 3.73839892e-06,
#        3.74145740e-06, 3.74451171e-06, 3.74756208e-06, 3.75060872e-06,
#        3.75365184e-06, 3.75669164e-06, 3.75972830e-06, 3.76276200e-06,
#        3.76579291e-06, 3.76882120e-06, 3.77184702e-06, 3.77487052e-06,
#        3.77789185e-06, 3.78091115e-06, 3.78392854e-06, 3.78694415e-06,
#        3.78995811e-06, 3.79297052e-06, 3.79598151e-06, 3.79899117e-06,
#        3.80199962e-06, 3.80500694e-06, 3.80801325e-06, 3.81101862e-06,
#        3.81402315e-06, 3.81702692e-06, 3.82003002e-06, 3.82303252e-06,
#        3.82603449e-06, 3.82903603e-06, 3.83203718e-06, 3.83503803e-06,
#        3.83803863e-06, 3.84103906e-06, 3.84403938e-06, 3.84703963e-06,
#        3.85003989e-06, 3.85304021e-06, 3.85604064e-06, 3.85904123e-06,
#        3.86204203e-06, 3.86504310e-06, 3.86804448e-06, 3.87104622e-06,
#        3.87404835e-06, 3.87705093e-06, 3.88005400e-06, 3.88305760e-06,
#        3.88606176e-06, 3.88906652e-06, 3.89207193e-06, 3.89507801e-06,
#        3.89808481e-06, 3.90109236e-06, 3.90410068e-06, 3.90710982e-06,
#        3.91011980e-06, 3.91313065e-06, 3.91614241e-06, 3.91915510e-06,
#        3.92216875e-06, 3.92518339e-06, 3.92819904e-06, 3.93121574e-06,
#        3.93423350e-06, 3.93725235e-06, 3.94027231e-06, 3.94329341e-06,
#        3.94631567e-06, 3.94933911e-06, 3.95236376e-06, 3.95538963e-06,
#        3.95841675e-06, 3.96144514e-06, 3.96447481e-06, 3.96750578e-06,
#        3.97053808e-06, 3.97357172e-06, 3.97660672e-06, 3.97964310e-06,
#        3.98268088e-06, 3.98572006e-06, 3.98876067e-06, 3.99180273e-06,
#        3.99484625e-06, 3.99789124e-06, 4.00093772e-06, 4.00398571e-06,
#        4.00703521e-06, 4.01008625e-06, 4.01313883e-06, 4.01619298e-06,
#        4.01924870e-06, 4.02230600e-06, 4.02536491e-06, 4.02842543e-06,
#        4.03148757e-06, 4.03455134e-06, 4.03761677e-06, 4.04068385e-06,
#        4.04375261e-06, 4.04682305e-06, 4.04989518e-06, 4.05296901e-06,
#        4.05604456e-06, 4.05912183e-06, 4.06220084e-06, 4.06528159e-06,
#        4.06836410e-06, 4.07144837e-06, 4.07453441e-06, 4.07762224e-06,
#        4.08071186e-06, 4.08380328e-06, 4.08689651e-06, 4.08999156e-06,
#        4.09308843e-06, 4.09618714e-06, 4.09928770e-06, 4.10239010e-06,
#        4.10549437e-06, 4.10860050e-06, 4.11170851e-06, 4.11481841e-06,
#        4.11793019e-06, 4.12104387e-06, 4.12415946e-06, 4.12727696e-06,
#        4.13039639e-06, 4.13351773e-06, 4.13664102e-06, 4.13976624e-06,
#        4.14289341e-06, 4.14602254e-06, 4.14915362e-06, 4.15228668e-06,
#        4.15542170e-06, 4.15855871e-06, 4.16169771e-06, 4.16483869e-06,
#        4.16798167e-06, 4.17112666e-06, 4.17427366e-06, 4.17742268e-06,
#        4.18057371e-06, 4.18372678e-06, 4.18688187e-06, 4.19003901e-06,
#        4.19319819e-06, 4.19635942e-06, 4.19952270e-06, 4.20268805e-06,
#        4.20585545e-06, 4.20902493e-06, 4.21219648e-06, 4.21537009e-06,
#        4.21854576e-06, 4.22172345e-06, 4.22490304e-06, 4.22808430e-06,
#        4.23126659e-06, 4.23444839e-06, 4.23762603e-06, 4.24079062e-06,
#        4.24392062e-06, 4.24696392e-06, 4.24979451e-06, 4.25210752e-06,
#        4.25316545e-06, 4.25118440e-06, 4.24184914e-06, 4.21472150e-06,
#        4.14455261e-06, 3.97026837e-06, 3.54413883e-06, 2.50882783e-06])


for itr in range(2,3):
    n = 2**itr
    h=(b-a)/n-1
    Ht2 = 0.5
    # print(h)
    c = 0.1
    X=np.linspace(a,b,n+1)
    # print("x",X)
    # Refs = np.zeros(len(X)-1)
    # for i in range(len(X)-1):
        # k = X[i]
        # index = np.where(refx == k)[0][0]
        # print(f"The index of {k} in ref X = {index}")
        # SS = ref[index]
        # print(SS)
        # Refs[i] = SS
        
    f = -Ht2*(1-(y/(1-alpha*y)))
    # print(f)
    f_fun = smp.lambdify(y,f)
    u0y = f/2
    uxy = f
    
    
    
    # intial guess for Quassilinirization 
    
    # y0x = 
    Y = np.zeros(n+1)
    Yi = np.zeros(n+1)
    mu = np.zeros(n+1)
    for i in range(0,n+1):
        mu[i] = ((1+c)**(X[i])-1)/(c)
        # w = 1
        # mu[i] = math.asin((c * math.cos(math.pi * i / n)) / math.asin(c))
    
    # print("mu",mu)
    # hb = np.zeros(n+1)
    hf = np.zeros(n+1)
    # for i in range(1,n+1):
    #     hb[i] = mu[i-1]-mu[i]
    for i in range(0,n):
        hf[i+1] = mu[i+1]-mu[i]
    # hf[0]=hf[1]
        # hf[n] = 0.17A
    # hf = np.zeros(n+1)
    # for i in range(n+1):
    #     hf[i] = h
    hb = hf
    # print("hb",hb)
    # print("hf",hf)
    
    M1 = 1/x
    #numerical values of M 
    M_fun = smp.lambdify(x,M1)
    M = np.zeros(n+1)
    
    for i in range(n+1):
        if X[i]!=0:
            M[i] = M_fun(X[i])
    
            # M[i] = 1/X[i]
        else:
            M[i]=0
    # print("M",M)    
      
    # Frist Derivative of M = Md1
    Md1 = smp.diff(M1,x)
    Md1_fun =smp.lambdify(x,Md1)
    # print(Md1)
    Md2 = smp.diff(Md1,x)
    Md2_fun =smp.lambdify(x,Md2)
    
    Md1 = np.zeros(n+1)
    Md2 = np.zeros(n+1)
    # print(Md2)
    for i in range(n+1):
        if X[i]!=0:
                Md1[i] = Md1_fun(X[i])
                Md2[i] = Md2_fun(X[i])
                # M[i] = 1/X[i]
        else:
            Md1[i]=0
            Md2[i]=0
    
        
    
    Y = np.zeros(n)
    Y0 = np.zeros(n)
    # print(Y)
    Y1 = np.zeros(n+1)
    Y2 = np.zeros(n+1)
    uy1 = np.zeros(n+1)
    uy2 = np.zeros(n+1)
    u1 = np.zeros(n+1)
    Vt1 = np.zeros(n+1)
    Vt2 = np.zeros(n+1)
    Ut1 = np.zeros(n+1)
    Ut2 = np.zeros(n+1)
    
    for i in range(n):
        Y0[i] = 1/(1+alpha)
    
    for l in range(5):
        for i in range(n): 
            # Y0[i] = 1/(1+alpha)
            # Y0 = np.array([0.0090162 ,0.01068306 ,0.01205378 ,0.00906355])
            
            Y[i] = -Ht2*(1-(Y0[i]/1-alpha*Y0[i]))
        
            uy1[i] = Ht2/(1-alpha*Y[i]**2)
            uy2[i] = 2*alpha*((1-alpha*Y[i])*Y2[i]+3*alpha*Y1[i]**2)/(1-alpha*Y[i])**4
            u1[i] = Ht2*(Y1[i])/(1-alpha*Y[i])**2
            # 
        # Y = Yi
        # Y = np.array([0.14206078,0.13517144,0.39992203,0.90909091])
        # print("1",Y)
        Ut = np.zeros(n)
        Vt = np.zeros(n)
        for i in range(1,n):
            Ut[i] = Ht2/(1-alpha*Y[i])**2
            Ut[0] = Ht2/(1-alpha*Y[0])**2*(0.5)
            Ut1[i-1] = Ut[i]-Ut[i-1]
            Ut2[i-1] = Ut1[i]-Ut1[i-1]
            # Ut1 = np.array([0.2455157,0,0,0,0,0])
            # Ut2 = np.array([-0.2455157,0,0,0,0,0])
            
            Vt[i] = f_fun(Y[i]) -Y[i]*Ht2/(1-alpha*Y[i])**2
            Vt[0] = (f_fun(Y[0]))/2 -Y[0]*Ht2/(1-alpha*Y[0])**2
            Vt1[i-1] = Vt[i]-Vt[i-1]
            Vt2[i-1] =  Vt1[i]-Vt1[i-1]
            # Vt1 = np.array([-0.27252252,0,0,0,0,0])
            # Vt2 = np.array([0.27252252,0,0,0,0,0])
            
        # # values of ai,bi,ci,di
        # ai = np.zeros(n)
        # bi= np.zeros(n)
        # ci= np.zeros(n)
        # di= np.zeros(n)
        # for i in range (0,n):
        #    ai[i] = -1 + (((hb[i] - hf[i+1]) / 3) * M[i]) + (((hb[i])**2 + (hf[i+1])**2 - hb[i] * hf[i+1]) / 12) * ((M[i])**2 +Ut[i]- Md1[i]) - ((hb[i] * hf[i+1]) / 6) * (M[i])**2 + ((hb[i] * (hf[i+1])**2 - (hb[i])**2 * hf[i+1]) / 24) * ((M[i])**3 + M[i]*Ut[i] - 2 * M[i] * Md1[i])
        #    bi[i] = ((hf[i+1] - hb[i]) / 3) * (Ut[i] - Md1[i])+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (M[i] * Md1[i] - M[i] * Ut[i] + 2 * Ut1[i] - Md2[i]) - M[i]+ (hb[i] * hf[i+1] / 6) * (M[i] * Ut[i] - M[i] * Md1[i])+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i]**2 * Md1[i] - M[i]**2 * Ut[i] + 2 * M[i] * Ut1[i] - M[i] * Md2[i])
        #    ci[i] = ((hf[i+1] - hb[i]) / 3) * (Ut1[i]) + ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Ut2[i] - M[i] * Ut1[i]) + (hb[i] * hf[i+1] / 6) * (M[i] * Ut1[i]) + ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Ut2[i] - M[i]**2 * Ut1[i]) + Ut[i]
        #    di[i] = ((hf[i+1] - hb[i]) / 3) * Vt1[i]+ ((hb[i]**2 + hf[i+1]**2 - hb[i] * hf[i+1]) / 12) * (Vt2[i] - M[i] * Vt1[i])+ (hb[i] * hf[i+1] / 6) * M[i] * Vt1[i]+ ((hb[i] * hf[i+1]**2 - hf[i+1] * hb[i]**2) / 24) * (M[i] * Vt2[i] - M[i]**2 * Vt1[i]) + Vt[i]
        
        # # for tridiagonal system 
        # p = np.zeros(n)
        # q = np.zeros(n)
        # r = np.zeros(n)
        # s = np.zeros(n)
        # rt = np.zeros(n)
        # st = np.zeros(n)
        
        # for i in range(0,n):
        #     p[i] = -2*ai[i]/(hb[i]*hf[i+1])
        #     q[i] = -bi[i]*((hb[i])**2 * (hf[i+1]**2))/(hb[i]*hf[i+1])
            
        # # for i in range(1,n-1):
        #     r[i] = -2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        #     rt[i] = 2*ai[i]/(hb[i]*(hb[i]+hf[i+1]))
        #     s[i] = bi[i]*hf[i+1]/(hb[i]*(hb[i]+hf[i+1]))
        #     st[i] = bi[i]*hb[i]/(hf[i+1]*(hb[i]+hf[i+1]))
        
        vi = np.zeros(n)
        wi = np.zeros(n)
        wti = np.zeros(n)
        # for i in range(n):
        #     vi[i] = p[i]+q[i]+ci[i]
        #     wti[i] = rt[i]-st[i]
        # for i in range(1,n):
        #     wi[i-1] = r[i]-s[i]
        
        for i in range(n):
            D = (mu[i+1]-mu[i])*(mu[i]-mu[i-1])/(mu[i+1]-mu[i-1])**2
            vi[i] = -12/D+2*((1/(mu[i]-mu[i-1])+1/(mu[i]-mu[i+1])))*M[i]+Ut[i]
            wti[i] = (12/D)*((mu[i+1]-mu[i])/(mu[i+1]-mu[i-1]))*M[i]*2*((mu[i]-mu[i-1])/(mu[i+1]-mu[i-1]))*((1/(mu[i+1]-mu[i]))+1/(mu[i+1]-mu[i-1]))
            
        for i in range(1,n):
            wi[i-1] = (12/D)*((mu[i+1]-mu[i])/(mu[i+1]-mu[i-1]))+M[i]*2*((mu[i]-mu[i+1])/(mu[i-1]-mu[i+1]))*((1/(mu[i-1]-mu[i]))+1/(mu[i-1]-mu[i+1]))
            
            
        
        # Sparse matrix A setup
        A = diags([wi, vi, wti], [-1,0,1], shape=(n, n)).toarray()
        A[0,1] = -12/D+2*((1/(mu[i]-mu[i-1])+1/(mu[i]-mu[i+1])))*M[i]+Ut[i]+(12/D)*((mu[i+1]-mu[i])/(mu[i+1]-mu[i-1]))*M[i]*2*((mu[i]-mu[i-1])/(mu[i+1]-mu[i-1]))*((1/(mu[i+1]-mu[i]))+1/(mu[i+1]-mu[i-1]))
        
        # A[0,1] = 
        R= np.zeros(n)
        for i in range(n):
            R[i] = -Vt[i] 
        S = np.linalg.solve(A,R)
        Y0 = S
        # print("S",Y0)
        
    # E = max(abs(Refs-Y0))
    # print(f"The Error on h = {hf} is {E}")
    
    
    # Adjust X to match the dimensions of S
    plt.figure(figsize=(8, 6))
    plt.plot(X[:n], S, marker='o', linestyle='-', color='b', label="Solution S")
    plt.xlabel("x")
    plt.ylabel("S (Solution)")
    plt.title("Numerical Solution")
    plt.legend()
    plt.grid()
    plt.show()
    
