#-----------------------------------------------------------------------------------------------------
# html.py
#-----------------------------------------------------------------------------------------------------
from html.parser import HTMLParser
#from html.entities import name2codepoint
#-----------------------------------------------------------------------------------------------------


escapers = {
    '&' : '&amp;',
    '<' : '&lt;', 
    '>' : '&gt;', 
    #' ' : '&nbsp;' 
}
def escape(s):
    t = str(s)
    for k,v in escapers.items():
        t = t.replace(k,v)        
    # no need to replace every space - remove ony
    # > starting, trailing and occuring more than once
    # > replace tab with 4 spaces,
    # > space with &nbsp
    # > newline with <br>
    return t

ascapers = {
    '"' : '&quot;',
    "'" : '&#39;', 
}
def ascape(s): # attribute-escape
    t = str(s)
    for k,v in ascapers.items():
        t = t.replace(k,v)
    return t


class HTAG:
    def __init__(self, open_type, tag, level, attributes, children) -> None:
        self.o, self.t, self.l, self.a, self.c = open_type, tag, level, attributes, children

    def __str__(self) -> str:
        return '~( LEV:|{}|  TAG:|{}|  TYPE:|{}|  ATR:|{}|  CHILD:|{}| )~'.format(self.l, (self.t if self.o!=0 else '!'), self.o, self.a, ( None if self.c is None else len(self.c) ) )  

    def __repr__(self) -> str:
        return self.__str__()

    def attrs(self):
        if self.a:
            return ' ' + ' '.join([ '{}="{}"'.format(_name, ascape(_value))  for _name, _value in self.a.items() ])
        else:
            return ''

    def relevel(self, lev=0):
        self.l = lev
        if self.c:
            for c in self.c:
                c.relevel(lev+1)

    """
    def code(self, f):
        #s = ''.join(['\t' for _ in range(lev)])
        if self.o == 0:
            f.write (escape(self.t)) # +nl
        else:
            f.write( '<' + self.t + self.attrs() + '>') # +nl
            if self.c:
                for c in self.c:
                    c.code(f)
            if self.o>0:
                f.write('</' + self.t + '>') # +nl
        
    def tree(self, p, lev=0):
        s = ''.join(['\t' for _ in range(lev)])
        p( s + str(self))
        if self.c:
            for c in self.c:
                c.tree(p, lev+1)
    """

class HARSER(HTMLParser):
    """ using python's inbuld html parser to convert html-> tree of elements """
    def __init__(self, strip_data) -> None:
        super().__init__(convert_charrefs=True)
        self.xstrip = strip_data
    

    # <--- implement --->
    def handle_decl(self, data):
        self.d.insert(0, HTAG(-1, data, 0, None, None) )

    def handle_starttag(self, tag, attrs):
        self.x.append( HTAG(-1, tag, self.lev, { k:v for k,v in attrs}, [])   ) #<--- assume its a open tag , tags have attr and child != None
        self.lev+=1

    def handle_data(self, data):
        if not data.isspace():
            self.x.append(HTAG(0,  data.strip() if self.xstrip else data, self.lev, None, None))  #<--- data tag has attr, child = None

    def handle_endtag(self, tag):
        L = [] 
        #print('Handle-end:', tag)
        #print('X', self.x)
        while self.x: # pop one by one
            ht = self.x.pop()
            #print('lev', self.lev, '~~pop:', ht)
            # first check its type
            if ht.o >= 0:
                L.append(ht)
            else:
                # now match level
                if ht.l == self.lev-1:
                    # now match tag
                    if ht.t == tag:
                        if L:
                            L.reverse()
                            ht.c.extend(L)
                        #else:
                        #    ht.c=None #<---------------- remove it make blank clid
                        ht.o=1
                        self.x.append(ht)
                        self.lev-=1
                        break
                    else: # inline tag
                        ht.c = None #<---------------- can't append to inline tag
                        L.append(ht)
                        self.lev-=1
        #print('end pop:', ht)
        #print('X', self.x)
        if not self.x:
            self.x.append(ht)
       
    # <---- one time parsing from html file ----> 
    def from_file(self, path, enc='utf-8'):
        self.x, self.d, self.lev  = [], [], 0
        with open(path, 'rt', encoding=enc, newline='\n') as f:
            self.feed( f.read()  )
        while self.d:
            self.x.insert(0, self.d.pop())
        del self.d, self.lev

        # relevel
        for x in self.x:
            x.relevel()
        return self.x

    def from_web(self, path, **args):
        import requests
        self.x, self.d, self.lev  = [], [], 0
        #with open(path, 'rt', encoding=enc, newline='\n') as f:
        r = requests.get( path, **args )
        r.close()
        self.feed( r.text  )
        while self.d:
            self.x.insert(0, self.d.pop())
        del self.d, self.lev

        # relevel
        for x in self.x:
            x.relevel()
        
        return self.x, r

    # <---- ignored ---->
    def handle_comment(self, data):
        print("Ignoring Comment:[{}]".format(data))
    def handle_entityref(self, name):
        """
        This method is called to process a named character reference of the form &name; 
        (e.g. &gt;), where name is a general entity reference (e.g. 'gt'). 
        This method is never called if convert_charrefs is True.
        
        c = chr(name2codepoint[name])
        print("Named ent:", c)
        """
        raise StopIteration('entity reference should be handled using convert_charrefs! :: [{}]'.format(name))
    def handle_charref(self, name):
        """
        This method is called to process decimal and hexadecimal numeric character references of the form &#NNN; and &#xNNN;. 
        For example, the decimal equivalent for &gt; is &#62;, whereas the hexadecimal is &#x3E;; in this case the method will receive '62' or 'x3E'. 
        This method is never called if convert_charrefs is True.
        
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        print("Num ent  :", c)
        """
        raise StopIteration('char reference should be handled using convert_charrefs! :: [{}]'.format(name))


def tree(htag, p, lev=0):
    s = ''.join(['\t' for _ in range(lev)])
    p( s + str(htag))
    if htag.c:
        for c in htag.c:
            tree(c, p, lev+1)

def code(htag, f, is_script=False):
    #s = ''.join(['\t' for _ in range(lev)])
    if htag.o == 0:
        f.write (  ( htag.t if is_script else escape(htag.t))  ) # +nl
        #f.write('\n')
    else:
        f.write( '<' + htag.t + htag.attrs() + '>') # +nl
        #f.write('\n')
        if htag.c:
            for c in htag.c:
                code(c, f, (htag.t.lower()=='script'))
        if htag.o>0:
            f.write('</' + htag.t + '>') # +nl #f.write('\n')

def from_file(path, strip_data=False, enc='utf-8'):
    """ parse a single html file """
    return HARSER(strip_data).from_file(path, enc)

def from_web(path, strip_data, **args):
    """ parse a external html page """
    return HARSER(strip_data).from_web(path, **args)

def to_file(hlist, path, is_script=False, enc='utf-8'):
    with open(path, 'wt', encoding=enc) as f:
        for h in hlist:
            code(h, f=f, is_script=is_script)
            f.write('\n')

#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:

    html element ref:    https://www.w3schools.com/tags/ref_byfunc.asp
    html attribute ref:  https://www.w3schools.com/tags/ref_attributes.asp
    

HELE_GLOBAL_ATR = {

    'accesskey'         : 'Specifies a shortcut key to activate/focus an element',
    'class'     	    : 'Specifies one or more classnames for an element (refers to a class in a style sheet)',
    'contenteditable'   : 'Specifies whether the content of an element is editable or not',
    'data-*'            : 'Used to store custom data private to the page or application',
    'dir'               : 'Specifies the text direction for the content in an element',
    'draggable'         : 'Specifies whether an element is draggable or not',
    'hidden'            : 'Specifies that an element is not yet, or is no longer, relevant',
    'id'                : 'Specifies a unique id for an element',
    'lang'              : 'Specifies the language of the element content',
    'spellcheck'        : 'Specifies whether the element is to have its spelling and grammar checked or not',
    'style'             : 'Specifies an inline CSS style for an element',
    'tabindex'          : 'Specifies the tabbing order of an element',
    'title'             : 'Specifies extra information about an element',
    'translate'         : 'Specifies whether the content of an element should be translated or not',
}


    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------

""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
