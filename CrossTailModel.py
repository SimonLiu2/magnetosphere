from OriginFieldModel import *
from mpl_toolkits.mplot3d import Axes3D
def override(func):
    return func
class CrossTailModel(OriginFieldModel):
    def __init__(self, tail_position=(-10,-40), tail_magnitude=1e-4, **kwargs):
        assert type(tail_position)==tuple and len(tail_position)==2, "tailposition must be a tuple of two values"
        assert type(tail_magnitude)==int or type(tail_magnitude)==float, "tailmagnitude must be a number"
        self.tail_position=tail_position
        self.tail_magnitude=tail_magnitude
        super().__init__(**kwargs)
        
    def tailField(self, y, z):
        dxdt = 0
        dydt = self.tail_magnitude*(np.arctan((y-self.tail_position[0])/z) - np.arctan((y-self.tail_position[1])/z))
        dzdt = self.tail_magnitude*(np.log(np.sqrt((y-self.tail_position[1])**2 + z**2)/np.sqrt((y-self.tail_position[0])**2 + z**2)))
        return np.array([dxdt, dydt, dzdt])
    
    @override
    def field_postive_raw(self, t, state, a):
        field1=super().field_postive_raw(t, state, a)
        field2=self.tailField(state[1],state[2])
        return field1+field2
    
    @override
    def field_negative_raw(self, t, state, a):
        field1=super().field_negative_raw(t, state, a)
        field2=self.tailField(state[1],state[2])
        return field1-field2
    
    @override
    def field_postive(self, t, state, a):
        return self.normalizer(self.field_postive_raw(t, state, a))
    
    @override
    def field_negative(self, t, state, a):
        return self.normalizer(self.field_negative_raw(t, state, a))