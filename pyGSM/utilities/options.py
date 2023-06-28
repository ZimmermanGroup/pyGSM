import collections


class Option(object):

    """ Class Option represents a key, value Option, with possible restrictions
        on type and value, and with documentation.
    """

    def __init__(
        self,
        key=None,
        value=None,
        required=False,
        allowed_types=None,
        allowed_values=None,
        doc="",
    ):
        """ Option constructor:

        Params/Members:
            key - the string value of the key of the option.
            value - the value of the option.
            required - True if the user is required to specify the option,
                False if the default value is acceptable. If required is True,
                an error will be raised when get_value is called if value is
                None. 
            allowed_types - array of allowed types of value. If this field is
                not None, a check will be performed on each call to set_value
                and get_value to ensure that value isinstance of at least one
                of the allowed_types.
            allowed_values - list of allowed values of value. If this field is
                not None, a check will be performed on each call to set_value 
                and get_value to ensure the value is one of the allowed_values.
            doc - string message providing helpful documentation about the
                option.
        """

        self.key = key
        self.value = value
        self.required = required
        self.allowed_types = allowed_types
        self.allowed_values = allowed_values
        self.doc = doc

    def get_value(self):
        """ Get value for this Option and check validity.

        Returns:
            value if the Option is in a valid state, else raises RuntimeError.
        """

        if self.required and self.value is None:
            raise RuntimeError("Option %s is required" % self.key)

        return self.value

    def set_value(self, value):
        """ Set value for this Option and check validity.

        Result:
            value is updated if Option is valid, else raises RuntimeError.
        """

        # Short-circuit if the value is None and the option is not required
        if value is None and not self.required:
            self.value = value
            return

        if self.allowed_types and not any(isinstance(value, x) for x in self.allowed_types):
            raise RuntimeError("Option %s must be one of allowed types: %s" % (
                self.key, self.allowed_types))
        if self.allowed_values and value not in self.allowed_values:
            raise RuntimeError("Option %s must be one of allowed values: %r" % (
                self.key, self.allowed_values))

        self.value = value

    def __str__(self):
        """ Return a string containing the full contents and documentation of this Option. """

        s = ''
        s += 'Option:\n'
        s += '  Key: %s\n' % self.key
        s += '  Value: %s\n' % self.value
        s += '  Required: %s\n' % self.required
        s += '  Allowed Types: %s\n' % self.allowed_types
        s += '  Allowed Values: %s\n' % self.allowed_values
        s += '  Doc: %s\n' % self.doc
        s += '\n'
        return s


class Options(object):

    """ Class Options represents a dict of key, value Option objects, including
        restrictions on type, value, etc. Users should interact only with the
        Options class - the Option class is used as internal data storage.

        Generally, codes declaring and using Options objects should first
        define the valid options, rules, and defaults by using the "add_option"
        method. Then, when the user wishes to set or get the values of specific
        options, a copy of the Options object should be provided for the user
        by calling the "copy" method of Options. 

        The underlying Option objects are stored in the options field, which is
        presently a collections.OrderedDict to remember the order of Option
        declaration. This incurs a 2x performance penalty vs. a standard dict
        object, so we may want to optimize the performance later.
    """

    def __init__(
        self,
        options=None,
    ):
        """ Options constructor.

        Params/Members:
            - options - dict of key -> Option
        """

        if options is None:
            self.options = collections.OrderedDict()
        else:
            self.options = options

    def keys(self):
        keys = []
        for opt in self.options:
            keys.append(opt)
        return keys

    def add_option(
        self,
        **kwargs
    ):
        """ Declare a new Option with possible default value, type and value
            rules, and documentation.

            Params: See Option constructor for valid kwargs
            Result: Options updated with new Option corresponding to key 
        """

        self.options[kwargs['key']] = Option(
            **kwargs
        )

    def get_option(
        self,
        key,
    ):
        """ Get the Option corresponding to key (useful for doc searching an debugging).

        Params:
            - key - string key of Option (raises RuntimeError if not in Options)
        Returns:
            - option - the explicit Option object (most users instead want the
              *value* of this object, which should be accessed through the
              __getitem__ method below).
        """

        if key not in self.options:
            raise ValueError("Key %s is not in Options" % key)
        return self.options[key]

    def __getitem__(
        self,
        key,
    ):
        """ Get the current value of Option corresponding to key, performing validity checks.

        Params:
            - key - string key of Option (raises RuntimeError if not in Options)
        Returns:
            - value - value of Option (raises RuntimeError if type, value or other validity error).
        """

        if key not in self.options:
            raise ValueError("Key %s is not in Options" % key)
        return self.options[key].get_value()

    def __setitem__(
        self,
        key,
        value,
    ):
        """ Set the value of Option corresponding to key, performing validity checks.

        Params:
            - key - string key of Option (raises RuntimeError if not in Options)
            - value - value of Option (raises RuntimeError if type, value or other validity error).
        Result:
            - Option value is updated if valid.
        """

        if key not in self.options:
            raise ValueError("Key %s is not in Options" % key)
        return self.options[key].set_value(value)

    def set_values(
        self,
        options,
    ):
        """ Set the values of multiple options. 

        Params:
            - options - dict of key, value pairs to set (calls __setitem__ once
              per key, value pair).
        Results:
            - Option values are updated if valid.
        """

        for k, v in options.items():
            self[k] = v
        return self

    def copy(self):
        """ Return a 1-level shallow copy of this Options object. This makes
            copies of all underlying Option objects so that changes to the new
            Options object will not affect the original Options object.
        """

        options2 = collections.OrderedDict()
        for k, v in self.options.items():
            options2[k] = Option(**v.__dict__)
        return Options(options=options2)

    def __str__(self):
        """ Return the string representations of all Option objects in this Options, in insertion order. """
        s = ''.join(str(v) for v in list(self.options.values()))
        return s


if __name__ == '__main__':

    import time

    print(" this demonstrates options")

    start = time.time()
    options1 = Options()
    for k in range(500):
        options1.add_option(
            key='size%d' % k,
            value=0,
            allowed_types=[int],
            allowed_values=[0, 1],
        )

    start = time.time()
    options2 = options1.copy()
    print('copy time %11.3E' % (time.time() - start))

    start = time.time()
    options3 = Options()
    options3.add_option(
        key='size',
        value=0,
        allowed_types=[int],
        allowed_values=[0, 1],
    )
    options4 = options3.copy()
    print('%11.3E' % (time.time() - start))

    start = time.time()
    options3 = Options()
    options3.add_option(
        key='size',
        value=0,
        allowed_types=[int],
        allowed_values=[0, 1],
    )
    options4 = options3.copy()
    print('%11.3E' % (time.time() - start))

    start = time.time()
    options3 = Options()
    options3.add_option(
        key='size',
        value=0,
        allowed_types=[int],
        allowed_values=[0, 1],
    )
    options4 = options3.copy()
    print('%11.3E' % (time.time() - start))

    options4.set_values({'size': 1})
    print(options4)
