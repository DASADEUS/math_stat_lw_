# -*- coding: utf-8 -*-

#Задание 3.1
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar
x = np.linspace(-3, 6, 1666)
y = gamma(x)
plt.plot(x, y, 'b', alpha=0.6, linewidth=4)
plt.xlim(-4, 6)
plt.ylim(-18, 20)
plt.grid()
plt.show()
min_f = minimize_scalar(gamma, bounds=(0,np.inf), method='brent')
print('x=',min_f.x)
print('min_function=',min_f.fun)

#Задание 3.2.1 функцию плотности вероятности
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import math
mu=0
var1=0.6
var2=3
var3=1
sigma1=math.sqrt(var1)
sigma2=math.sqrt(var2)
sigma3=math.sqrt(var3)
x1=np.linspace(mu-3*sigma1,mu+3*sigma1,100)
x2=np.linspace(mu-3*sigma2,mu+3*sigma2,100)
x3=np.linspace(mu-3*sigma3,mu+3*sigma3,100)
plt.plot(x1,st.norm.pdf(x1,mu,sigma1),label = u'mu=0,var=0.6')
plt.plot(x2,st.norm.pdf(x2,mu,sigma2),label = u'mu=0,var=3')
plt.plot(x3,st.norm.pdf(x3,mu,sigma3),label = u'mu=0,var=1')
plt.grid()
plt.legend(loc='upper right')

#Задание 3.2.2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import math
mu1=-1
mu2=1
mu3=3
var=3
sigma=math.sqrt(var)
x1=np.linspace(mu1-3*sigma,mu1+3*sigma,100)
x2=np.linspace(mu2-3*sigma,mu2+3*sigma,100)
x3=np.linspace(mu3-3*sigma,mu3+3*sigma,100)
plt.plot(x1,st.norm.pdf(x1,mu1,sigma),label = u'mu1=-1,var=3')
plt.plot(x2,st.norm.pdf(x2,mu2,sigma),label = u'mu1=1,var=3')
plt.plot(x3,st.norm.pdf(x3,mu3,sigma),label = u'mu1=3,var=3')
plt.grid()
plt.legend(loc='upper right')

#Задание 3.3
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
pros=np.linspace(0,20,1000)
plt.plot(pros,chi2.pdf(pros,2),label = u'Степень свободы 2')
plt.plot(pros,chi2.pdf(pros,3),label = u'Степень свободы 3')
plt.plot(pros,chi2.pdf(pros,5),label = u'Степень свободы 5')
plt.plot(pros,chi2.pdf(pros,10),label = u'Степень свободы 10')
plt.grid()
plt.legend(loc='upper right')
plt.axvline(x=10.191027908788493, ymin=0, ymax=1, color='red')
print(chi2.sf(6,3))
print(1-chi2.cdf(6,3))
print(chi2(5).sf(6))

#Задание 3.3B
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(0,50,100)
#chi3.sf
a=chi2.cdf(10.191027908788493,5)
plt.plot(pros,chi2.cdf(pros,20))
plt.axvline(x=10.9, ymin=0, ymax=1, color='red')
plt.grid()

print('Fx2,20(10.9)=',a)
print('P(x2,20>10.9)=',1-a)

import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(0,50,100)
a=chi2.cdf(28.9,20)
plt.plot(pros,chi2.cdf(pros,20))
plt.axvline(x=28.9, ymin=0, ymax=1, color='red')
plt.grid()

print('P(x2,20<28.9)=',a)

#Задание 3.3B
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(0,50,100)
a=chi2.cdf(8.26,25)
b=chi2.cdf(31.4,25)
plt.plot(pros,chi2.cdf(pros,25))
plt.axvline(x=8.26, ymin=0, ymax=1, color='red')
plt.axvline(x=31.4, ymin=0, ymax=1, color='red')
plt.grid()

print('P(x2,20=8.26)=',a)
print('P(x2,20=31.4)=',b)
print('P(8.26<x2,20<31.4)=',b-a)

#Задание 3.3Г
x1=chi2.isf(0.1,5)
x2=chi2.isf(0.07,5)
x3=chi2.isf(0.95,18)
x4=chi2.isf(0.99,80)

print('x2,0.1(5)=',x1)
print('x2,0.01(100)=',x2)
print('x2,0.95(18)=',x3)
print('x2,0.99(80)=',x4)

#Задание 3.4
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import scipy.stats as st
import math
pros=np.linspace(-4,4,100)
plt.plot(pros,t.pdf(pros,1),label = u'n=1')
plt.plot(pros,t.pdf(pros,2),label = u'n=2')
plt.plot(pros,t.pdf(pros,4),label = u'n=4')
plt.grid()
plt.legend(loc='upper right')

#Задание 3.4Г
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(0,50,100)
#chi2.sf
a=t.cdf(2.23,10)
plt.plot(pros,t.cdf(pros,10))
plt.axvline(x=2.23, ymin=0, ymax=1, color='red')
plt.grid()
print('P(t10<2.23)=',a)

#Задание 3.4Г
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(-5,50,100)
#chi2.sf
a=t.cdf(-2.23,10)
plt.plot(pros,t.cdf(pros,10))
plt.axvline(x=-2.23, ymin=0, ymax=1, color='red')
plt.grid()
print('P(t10>-2.23)=',1-a)

#Задание 3.4Г
import matplotlib.pyplot as plt
from scipy.stats import chi2
import numpy as np

pros=np.linspace(-5,50,100)
#chi2.sf
a=t.cdf(2.23,10)
b=t.cdf(-2.23,10)
plt.plot(pros,t.cdf(pros,10))
plt.axvline(x=2.23, ymin=0, ymax=1, color='red')
plt.axvline(x=-2.23, ymin=0, ymax=1, color='red')
plt.grid()
print('P(|t10|<2.23)=',a-b)

#Задание 3.4Г
import matplotlib.pyplot as plt
from scipy.stats import t
import numpy as np

pros=np.linspace(-5,50,100)
#chi2.sf
a=t.cdf(3.17,10)
b=t.cdf(-1.31,10)
plt.plot(pros,t.cdf(pros,10))
plt.axvline(x=3.17, ymin=0, ymax=1, color='red')
plt.axvline(x=-1.31, ymin=0, ymax=1, color='red')
plt.grid()
print('P(-1.31<t10<3.17)=',a-b)

#Задание 3.4Г
import matplotlib.pyplot as plt
from scipy.stats import t
import numpy as np

pros=np.linspace(0,50,100)
#chi2.sf
a=t.cdf(1.96,82)
plt.plot(pros,t.cdf(pros,82))
plt.axvline(x=1.96, ymin=0, ymax=1, color='red')
plt.grid()
print('P(t82>1.96)=',1-a)

#Задание 3.4Г
x1=t.isf(0.005,33)
x2=t.isf(0.1,23)
x3=t.isf(0.05,100)
x4=t.isf(0.025,math.inf)

print('t,0.005(33)=',x1)
print('t,0.1(23)=',x2)
print('t,0.05(100)=',x3)
print('t,0.025(00)=',x4)

#Задание 3.5
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f
import scipy.stats as st
import math
pros=np.linspace(0,3,100)
plt.plot(pros,f.pdf(pros,2,10),label = u'n=2,m=10')
plt.plot(pros,f.pdf(pros,5,10),label = u'n=5,m=10')
plt.plot(pros,f.pdf(pros,20,10),label = u'n=20,m=10')
plt.grid()
plt.legend(loc='upper right')

#Задание 3.5Г
import matplotlib.pyplot as plt
from scipy.stats import f
import numpy as np

pros=np.linspace(-5,10,100)
#chi2.sf
a=f.cdf(3.24,3,16)
b=f.cdf(-3.24,3,16)
plt.plot(pros,f.cdf(pros,3,16))
plt.axvline(x=3.24, ymin=0, ymax=1, color='red')
plt.axvline(x=-3.24, ymin=0, ymax=1, color='red')
plt.grid()
print('P(|F3;16|<3.24)=',a-b)

#Задание 3.5Г
import matplotlib.pyplot as plt
from scipy.stats import f
import numpy as np

pros=np.linspace(0,5,100)
#chi2.sf
a=f.cdf(1.3,35,100)
plt.plot(pros,f.cdf(pros,35,100))
plt.axvline(x=1.3, ymin=0, ymax=1, color='red')
plt.grid()
print('P(|F3;16|<3.24)=',1-a)

#Задание 3.5Г
x1=f.isf(0.05,3,7)
x2=f.isf(0.05,7,3)
x3=f.isf(0.025,5,20)
x4=f.isf(0.05,300,800)

print('F,0.05(3,7)=',x1)
print('F,0.05(7,3)=',x2)
print('F,0.025(5,20)=',x3)
print('F,0.05(300,800)=',x4)

#Задание 3.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nct
import scipy.stats as st
import math
pros=np.linspace(-6,6,100)
plt.plot(pros,nct.pdf(pros,25,-2),label = u'o=-2,v=25')
plt.plot(pros,nct.pdf(pros,25,0),label = u'o=0,v=25')
plt.plot(pros,nct.pdf(pros,25,3),label = u'o=3,v=25')

plt.grid()
plt.legend(loc='upper right')

#Задание 3.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nct
import scipy.stats as st
import math
pros=np.linspace(-2,6,100)
plt.plot(pros,nct.pdf(pros,0.5,2),label = u'o=2,v=0.5')
plt.plot(pros,nct.pdf(pros,3,2),label = u'o=2,v=3')
plt.plot(pros,nct.pdf(pros,10,2),label = u'o=2,v=10')

plt.grid()
plt.show()

#Задание 3.6B
x1=nct.ppf(0.2,1,2)
x2=nct.ppf(0.4,1,2)
x3=nct.ppf(0.6,1,2)
x4=nct.ppf(0.8,1,2)

print('f,1;2(0.2)=',x1)
print('f,1;2(0.4)=',x2)
print('f1;2(0.6)=',x3)
print('f,1;2(0.8)=',x4)

#Задание 3.6Г
x1=nct.sf(2.262,9,0.632)
print('P(T>2.262),T-t(9;0,632)=',x1)

#Задание 3.8
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2
import scipy.stats as st
import math
pros=np.linspace(0,30,100)
plt.plot(pros,ncx2.pdf(pros,6,0.5),label = u'v=6,л=0.5')
plt.plot(pros,ncx2.pdf(pros,6,2),label = u'v=6,л=2')
plt.plot(pros,ncx2.pdf(pros,6,7),label = u'v=6,л=7')

plt.grid()
plt.legend(loc='upper right')

#Задание 3.8B
v=6
q=0.2
x1=nct.ppf(q,v,0.5)
x2=nct.ppf(q,v,2)
x3=nct.ppf(q,v,7)
q=0.4
x4=nct.ppf(q,v,0.5)
x5=nct.ppf(q,v,2)
x6=nct.ppf(q,v,7)
q=0.6
x7=nct.ppf(q,v,0.5)
x8=nct.ppf(q,v,2)
x9=nct.ppf(q,v,7)
q=0.8
x10=nct.ppf(q,v,0.5)
x11=nct.ppf(q,v,2)
x12=nct.ppf(q,v,7)
print(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)

#Задание 3.9
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncf
import scipy.stats as st
import math
pros=np.linspace(0,10,100)
plt.plot(pros,ncf.pdf(pros,2,10,0.5))
plt.plot(pros,ncf.pdf(pros,2,10,5))
plt.plot(pros,ncf.pdf(pros,2,10,10))
plt.grid()