#-----------------------------------------------------------------------------------------------------
# printer.py
#-----------------------------------------------------------------------------------------------------
import datetime
#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Printing functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="[\n\t", sep="\n\t", end="\n]"):
    """ returns a string representation of an array/list for printing """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep="\t:\t", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-="+sep+"DICT: "+caption+sep+"=-=-=-=-==-=-=-=-="+sep
    for i in arr:
        res+=str(i) + cep + str(arr[i]) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def showD(x, sep="\n", cep='\t:\t', P = print):
    """ prints x.__dict__ using strD() """
    P(strD(x.__dict__, sep=sep, cep=cep, caption=str(x.__class__)))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(x, cep='\t\t:', sw='__', ew='__', P = print):
    """ Note: 'sw' can accept tuples """
    for d in dir(x):
        if not (d.startswith(sw) or d.endswith(ew)):
            v = ""
            try:
                v = getattr(x, d)
            except:
                v='?'
            P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def showX(x, cep='\t\t:',P = print):
    """ same as showx but shows all members, skip startswith test """
    for d in dir(x):
        v = ""
        try:
            v = getattr(x, d)
        except:
            v='?'
        P(d, cep, v)

class SHOW:
    def __init__(self, start='\n', sep='\n', cep='\t:\t', end='\n', sw='__', ew='__', P = print) -> None:
        self.start, self.sep, self.cep, self.end = start, sep, cep, end
        self.sw, self.ew = sw, ew
        self.P = P
    def __call__(self, x):
        self.P(self.start)

        if (self.sw or self.ew):# use all attr
            for d in dir(x):
                if not (d.startswith(self.sw) or d.endswith(self.ew)):
                    v = ""
                    try:
                        v = getattr(x, d)
                    except:
                        v='?'
                    self.P(d, self.cep, v)
                    self.P(self.sep)

        else:
            for d in dir(x):
                v = ""
                try:
                    v = getattr(x, d)
                except:
                    v='?'
                self.P(d, self.cep, v)
                self.P(self.sep)


        self.P(self.end)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Time-stamping functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def now(format='%Y-%m-%d %H:%M:%S::%f'):
    """ formated time stamp 
        %a	Weekday, short version	Wed	
        %A	Weekday, full version	Wednesday	
        %w	Weekday as a number 0-6, 0 is Sunday	3	
        %d	Day of month 01-31	31	
        %b	Month name, short version	Dec	
        %B	Month name, full version	December	
        %m	Month as a number 01-12	12	
        %y	Year, short version, without century	18	
        %Y	Year, full version	2018	
        %H	Hour 00-23	17	
        %I	Hour 00-12	05	
        %p	AM/PM	PM	
        %M	Minute 00-59	41	
        %S	Second 00-59	08	
        %f	Microsecond 000000-999999	548513	
        %z	UTC offset	+0100	
        %Z	Timezone	CST	
        %j	Day number of year 001-366	365	
        %U	Week number of year, Sunday as the first day of week, 00-53	52	
        %W	Week number of year, Monday as the first day of week, 00-53	52	
        %c	Local version of date and time	Mon Dec 31 17:41:00 2018	
        %C	Century	20	
        %x	Local version of date	12/31/18	
        %X	Local version of time	17:41:00	
        %%	A % character	%	
        %G	ISO 8601 year	2018	
        %u	ISO 8601 weekday (1-7)	1	
        %V	ISO 8601 weeknumber (01-53)	01
    """
    return datetime.datetime.strftime(datetime.datetime.now(), format)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strU(sep=''):
    """ formated time stamp based UID """
    return datetime.datetime.strftime(datetime.datetime.now(), sep.join(["%Y","%m","%d","%H","%M","%S","%f"]))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strUx(start='', sep='', end=''):
    """ xtended formated time stamp based UID """
    return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(["%Y","%m","%d","%H","%M","%S","%f"])) + end


#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Dummy objects
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
OBJECT = lambda members: type('object', (object,), members)() #<---- create a dummy object on the fly (members is a dictionary)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class O:
    """ dummy object that mimics a dict """
    def __init__(self, **config_dict):
        for key, val in config_dict.items():
            if not (hasattr(self, key)):
                setattr(self, key, val) # dict[key] = getattr(self, key)
            else:
                print('Warning: cannot attribute with name [{}]! Skipping.'.format(key))
    def __bool__(self):
        return True if self.__dict__ else False
    def __len__(self):
        return len(self.__dict__)
    def __call__(self, key=None):
        if key:
            return key, self.__dict__[key]
        else:
            return self.__dict__.items()
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


from math import floor
#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num:int, base:int, digs:int) -> list:
    """ convert base-10 integer (num) to base(base) array of fixed no. of digits (digs) """
    res = [ 0 for _ in range(digs) ]
    q = num
    for i in range(digs): # <-- do not use enumerate plz
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(num:list, base:int) -> int:
    """ convert array from given base to base-10  --> return integer """
    res = 0
    for i,n in enumerate(num):
        res+=(base**i)*n
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class RCONV:
    def __init__(self,Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm