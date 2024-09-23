import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
#由于该模型的计算量较大，所以有必要时可以采用向量化的方式来进行计算
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('Time Cost:',end-start)
        return result
    return wrapper

class OriginFieldModel():
    def __init__(self,a=10,rate=0.01,demo=True):
        self.a = a
        self.rate = rate
        self.t_span = [0, 100]
        self.t_eval = np.linspace(0, 100, int(1e4))
        self.benchmark = self.a*2+3
        self.demo = demo
        self.xlim = (-self.benchmark, self.benchmark)
        self.ylim = (-self.benchmark, self.benchmark)
        self.zlim = (-self.benchmark, self.benchmark)
    def set_lim(self,xlim=None,ylim=None,zlim=None):
        if xlim is not None:
            if not (isinstance(xlim, tuple) and len(xlim) == 2):
                raise ValueError('xlim should be a tuple of length 2')
            self.xlim = xlim
        if ylim is not None:
            if not (isinstance(ylim, tuple) and len(ylim) == 2):
                raise ValueError('ylim should be a tuple of length 2')
            self.ylim = ylim
        if zlim is not None:
            if not (isinstance(zlim, tuple) and len(zlim) == 2):
                raise ValueError('zlim should be a tuple of length 2')
            self.zlim = zlim
    def initialize_state(self):
        if self.demo:
            theta = np.arange(0,np.pi/6,np.pi/30).reshape(-1, 1)
            phi = np.arange(0,2*np.pi,np.pi/4).reshape(1, -1)
            r=1.01
            x = r*np.sin(theta)@np.cos(phi)
            y = r*np.sin(theta)@np.sin(phi)
            z = r*np.cos(theta)*np.ones_like(phi)
            self.x=x.flatten()
            self.y=y.flatten()       
            self.z=z.flatten()
        else:
            theta = np.arange(0,np.pi/2,np.pi/60).reshape(-1, 1)
            phi = np.arange(0,2*np.pi,np.pi/4).reshape(1, -1)
            r=1.01
            x = r*np.sin(theta)@np.cos(phi)
            y = r*np.sin(theta)@np.sin(phi)
            z = r*np.cos(theta)*np.ones_like(phi)
            self.x=x.flatten()
            self.y=y.flatten()       
            self.z=z.flatten()
    def normalizer(self,vec):
        assert type(vec)==np.ndarray, 'The input should be a numpy array'
        return vec/np.linalg.norm(vec)
    def field_postive_raw(self,t,state,a):
        x, y, z = state
        r1 = np.sqrt(x**2 + y**2 + z**2)
        r2 = np.sqrt(x**2 + (y-2*a)**2 + z**2)
        dxdt = 3*x*z/r1**5 + 3*x*z/r2**5
        dydt = 3*y*z/r1**5 + 3*(y-2*a)*z/r2**5
        dzdt = 3*z**2/r1**5 + 3*z**2/r2**5 - 1/r2**3 - 1/r1**3
        return np.array([dxdt, dydt, dzdt])
    def field_negative_raw(self,t,state,a):
        x, y, z = state
        r1 = np.sqrt(x**2 + y**2 + z**2)
        r2 = np.sqrt(x**2 + (y-2*a)**2 + z**2)
        dxdt = 3*x*z/r1**5 + 3*x*z/r2**5
        dydt = 3*y*z/r1**5 + 3*(y-2*a)*z/r2**5
        dzdt = 3*z**2/r1**5 + 3*z**2/r2**5 - 1/r2**3 - 1/r1**3
        return np.array([-dxdt, -dydt, -dzdt])
    
    def field_postive(self,t,state,a):
        vec=self.field_postive_raw(t,state,a)
        return self.normalizer(vec)
    def field_negative(self,t,state,a):
        vec=self.field_negative_raw(t,state,a)
        return self.normalizer(vec)
    
    def event1(self,t,state,a):
        x, y, z = state
        return np.sqrt(x**2 + y**2 + z**2)-1 
    def event2(self,t,state,a):
        x, y, z = state
        return max(abs(x),abs(y),abs(z))-self.benchmark 
    event1.terminal = True
    event1.direction = 0
    event2.terminal = True
    event2.direction = 0
    @timer
    def solver(self):
        self.initialize_state()
        x = self.x
        y = self.y
        z = self.z
        series = []
        i=0
        origin_len = len(x)
        while i<len(x):
            B_vec = np.array([x[i],y[i],z[i]])
            if B_vec[2]>0:
                solution = solve_ivp(self.field_postive, self.t_span, B_vec, t_eval=self.t_eval, \
                                       args=(self.a,),events=[self.event1,self.event2])
            else:
                solution = solve_ivp(self.field_negative, self.t_span, B_vec, t_eval=self.t_eval, \
                                       args=(self.a,),events=[self.event1,self.event2])
            if len(solution.t_events[0])==0 and i<origin_len:
                x = np.append(x,x[i])
                y = np.append(y,y[i])
                z = np.append(z,-z[i])
            series.append(solution.y)
            i+=1
        print('The number of lines:',len(x))
        return series
    def plot(self):
        series = self.solver()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(series)):
            x = series[i][0]
            y = series[i][1]
            z = series[i][2]
            if np.sqrt(x[-1]**2+y[-1]**2+z[-1]**2)>self.benchmark:
                ax.plot(x,y,z,color='r',linewidth=0.8)
            else:
                ax.plot(x,y,z,color='b',linewidth=0.8)
        # 创建球体的网格数据
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        # 在三维图的正中心添加一个半径为1的地球
        ax.plot_surface(x, y, z, color='g', alpha=0.6)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_zlim(self.zlim)
        ax.set_xlabel('X')  
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()