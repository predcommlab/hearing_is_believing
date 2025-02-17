#import serial
from psychopy import visual, core, parallel


class Simulated:
    '''
    '''

    def __init__(self):
        '''
        '''

        pass
    
    def setData(self, code: str):
        '''
        '''

        print(f'trigger::Simulated::setData(): Received `code`={code}.')



class Port:
    '''
    '''

    def __init__(self, win: visual.Window, name: str = '0x0278', simulate: bool = False, L = 5e-3):
        '''
        '''

        self.L = L
        self.win = win
        self.on = False

        if simulate: self.port = Simulated(); print('WARNING: Using _simulated_ serial port.')
        #else: self.port = serial.Serial(name, timeout = 5.0, write_timeout = 5.0)
        else: self.port = parallel.ParallelPort(address = '0xCFF8')
    
    def send(self, code: str):
        '''
        '''

        self.port.setData(int(code))
        core.wait(self.L)
        self.port.setData(int(str.encode('0')))

        '''
        self.win.callOnFlip(self.port.write, str.encode(code))
        core.wait(self.L)
        self.win.callOnFlip(self.port.write, str.encode('0'))
        '''