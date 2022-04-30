#-----------------------------------------------------------------------------------------------------
# struct.py
#-----------------------------------------------------------------------------------------------------
import os
GLOBAL_CTRL_CHAR = {

    'SOH'   : 1,    # start of header
    'STX'   : 2,    # start of text
    'ETX'   : 3,    # end of text
    'EOT'   : 4,    # end of transmission
    'ETB'   : 23,   # end of transmission block
    'EOM'   : 25,   # end of medium

    'DLE'   : 16,
    'DC1'   : 17,
    'DC2'   : 18,
    'DC3'   : 19,
    'DC4'   : 20,

    'FS'   : 28,
    'GS'   : 29,
    'RS'   : 30,    # Record Seperator      
    'US'   : 31,    # Unit Seperator        

    'ENQ'   : 5,
    'ACK'   : 6,
    'NAK'   : 21,
    'BEL'   : 7,
    'CAN'   : 24,
    'SYN'   : 22,
    'SUB'   : 26,
    'ESC'   : 27,
    'SI'   : 14,
    'SO'   : 15,
}
#-----------------------------------------------------------------------------------------------------



#------------------------------------
# indexed table of records - iTOR data structure
#------------------------------------
class iTOR:
    """ (i)ndexed (T)able (O)f (R)ecords 
    
        its just a,         list of(    records of(     units   )))
        which basically,    list of(    list of(        strings )))
        an iTOR is represented using 2 files - data file and index file
    """
    #<< class variables
    GLOBAL_US =     GLOBAL_CTRL_CHAR['US']
    GLOBAL_RS =     GLOBAL_CTRL_CHAR['RS']
    def create(path, index_ext='.index'):
        return iTOR(path, os.path.join(os.path.dirname(path), os.path.basename(path) + index_ext) )


    #<< dunder methods      ----------------------------------------------------------
    def __init__(self, data_file_path, index_file_path) -> None:
        #assert(index.lower.endswith('.npy'))
        self.path= data_file_path
        self.index = index_file_path
        self.US = bytes([int(self.GLOBAL_US)]) # unit seperator
        self.RS = bytes([int(self.GLOBAL_RS)]) # record seperator
        self.mode = '' #<-- is closed
    def __iter__(self):
        return self.iterR()
    def __len__(self):
        return self.len()


    #<< open method
    def _iopen(self):
        with open(self.index, 'rt', encoding='utf-8', newline='\n') as f:
            res = [ ([ int(y) for y in x[0:-1].split(' ')]  if len(x)>1 else [])  for x in f.readlines() ]
        return res
    def _openi(self):
        if self.mode!='w':
            try:
                i = self._iopen()
                self.i = i if i else []
            except FileNotFoundError:
                self.i = []
        else:
            self.i = []
        self.i.append([(self.i[-1][-1]+1) if self.i else 0])
    def open(self, mode='w'): 
        """ opens the list for appending, mode should be either w/a """
        if not self.mode:
            self.mode=mode
            self.f = open(self.path, mode=self.mode+'b')
            self._openi()
        else:
            print('! Already open in [{}] mode !'.format(self.mode))


    #<< close method        ----------------------------------------------------------
    def _iclose(self):
        with open(self.index, 'wt', encoding='utf-8', newline='\n') as f:
            for i in self.i:
                f.write( ' '.join([ str(d) for d in i ]) + '\n' )
            del self.i
        return self.index
    def _closei(self):
        #assert(not self.i.pop()) #<----- means that some row might be open.... close it?
        lR = self.i.pop()
        if len(lR)>1:
            print('! Last Record was not closed - ignoring data: [{}] !'.format(lR))
        self._iclose()
    def close(self):
        """ closes the list, should be called to free resources """
        if self.mode:
            self.f.close()
            self._closei()
            self.mode=''
        else:
            print('! Already closed !')


    #<< write method        ----------------------------------------------------------
    def _uwrite(self, *units):
        for unit in units:
            self.f.write(str(unit).encode('utf-8'))
            self.f.write(self.US)
            self.i[-1].append( self.f.tell() )
    def _uclose(self):
        self.f.write(self.RS)
        self.i.append([self.f.tell()])
    def write(self, *rows): # rows are records
        """ args = multiple rows, writes multiple rows at once
                assumes that each item in args is a row itself - row should be iterable """
        for row in rows:
            self.writeR(*row)
    def writeR(self, *units):
        """ args = multiple units in a row, writes one full row at once
                assumes that each item in arg is a item(unit) in this row """
        self._uwrite(*units)
        self._uclose()
    def writeU(self, *units, close=False):
        """ args = multiple cells in a row, 
                assumes that each item in arg is a item(cell) in this row """
        self._uwrite(*units)
        self._uclose() if close else None
        

    #<< read method     ----------------------------------------------------------
    def readU(self, row, unit):
        a, b = self.i[row][unit], self.i[row][unit+1]
        self.f.seek(a)
        return self.f.read(b - a - 1).decode('utf-8')
    def readR(self, row):
        a, b = self.i[row][0], self.i[row][-1]
        self.f.seek(a)
        return [ d.decode('utf-8') for d in self.f.read(b - a - 1).split(self.US) ]
    def read(self, *rows):
        return [ self.readR(row) for row in rows ]
    def readA(self):
        return self.read(*range(len(self)))
    

    #<< generator method ----------------------------------------------------------
    def iterU(self):
        if not self.mode:
            self.open('r')
        else:
            if self.mode!='r':
                raise Exception('! Already open in [{}] mode - close it first!'.format(self.mode))
        ptr = 0
        self.f.seek(ptr)
        temp= self.i.pop()
        for ii in self.i:
            assert(ptr==ii[0]) #<---- indicates errors in file
            for index in ii[1:]:
                i =  index - ptr - 1
                buffer = self.f.read(i)
                _ = self.f.read(1)
                ptr += (i +1)
                yield buffer.decode('utf-8')
            _ = self.f.read(1)
            ptr += 1
            yield None #<--- None to mark end of row
        del ptr #<-- clean up
        self.i.append(temp)
        self.close()
    def iterR(self):
        if not self.mode:
            self.open('r')
        else:
            if self.mode!='r':
                raise Exception('! Already open in [{}] mode - close it first!'.format(self.mode))
        ptr = 0
        self.f.seek(ptr)
        temp= self.i.pop()
        for ii in self.i:
            assert(ptr==ii[0]) #<---- indicates errors in file
            row = []
            for index in ii[1:]:
                i =  index - ptr - 1
                buffer = self.f.read(i)
                _ = self.f.read(1)
                ptr += (i +1)
                row.append(buffer.decode('utf-8'))
            _ = self.f.read(1)
            ptr += 1
            yield row #<--- None to mark end of row
        del ptr #<-- clean up
        self.i.append(temp)
        self.close()


    #<< info method     ----------------------------------------------------------
    def len(self):
        return len(self.i)-1 if self.mode else 0
    def count(self, i):
        return len(self.i[i])-1 if self.mode else -1
    def info(self, p=print):
        m,l = self.mode, self.len()
        p('Mode[{}]/Len:[{}]'.format(m,l))
        if m:
            p('Count:')
            for i in range(l):
                p('\t_{}_\t:\t[{}]'.format(i, self.count(i)))

    def save(path, data):
        itor = iTOR.create(path)
        itor.open()
        itor.write(*data)
        itor.close()
    def load(path):
        itor = iTOR.create(path)
        return list(itor.iterR())


#------------------------------------
# multi context manager for iTOR
#------------------------------------
class MCM:
    """ multi context-manager """
    def from_lot(*lot):
        ds, mode = [], []
        for d,m in lot:
            ds.append(d)
            mode.append(m)
        return MCM(ds, mode)
    def from_list(dsL, mode):
        return MCM(dsL, [mode for _ in range(len(dsL))] )
    def from_lists(dsL, modeL):
        return MCM(dsL, modeL)

    def __init__(self, ds, mode) -> None:
        self.ds, self.mode = ds, mode
    def __enter__(self):
        for ds, mode in zip(self.ds, self.mode):
            ds.open(mode)
        return self.ds
    def __exit__(self, exc_type, exc_val, exc_tb):
        for ds in self.ds:
            ds.close()
        return True




#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
