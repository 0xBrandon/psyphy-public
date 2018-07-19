#############################################################################
# MIT License
#
# Copyright (c) 2018 Brandon RichardWebster
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#############################################################################

import uuid

def static_vars(**kwargs):
    '''
        Store a static variables in a function [1].

        Parameters:
        -----------
        **kwargs :  allows you to pass keyworded variable length of arguments
                    to a function.

        Example:
        --------
        @static_vars(counter=0)
        def foo():
            foo.counter += 1
            print "Counter is %d" % foo.counter

        [1] See http://stackoverflow.com/questions/279561/what-is-the-python-
                equivalent-of-static-variables-inside-a-function for original
            forum post and further description of function.
    '''

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def upath(ext, pre='/tmp/'):
    '''
        Creates a unique, random file path using uuid.uuid4()[1].

        Parameters:
        -----------
        ext :   str
                Specify the extension of the file.
                e.g., .jpg, .txt.

        pre :   str, optional
                Specify the prefix for the unique path.
                e.g., '/tmp/', '/home/foobar/'.

        Returns:
        --------
        out :   str
                A unique filepath.

        [1]     See standard Python uuid for further description of uuid4().
    '''
    return ''.join([pre, str(uuid.uuid4()), ext])


class OutOfBoundsError(Exception):
    pass
