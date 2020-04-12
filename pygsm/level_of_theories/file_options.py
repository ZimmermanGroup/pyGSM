from collections import namedtuple, defaultdict, OrderedDict
from re import sub
from ast import literal_eval as leval

class File_Options(object):
    """ Class file_options allows parsing of an input file
    """
    def __init__(self, input_file):
        self.Documentation = OrderedDict()
        self.UserOptions = OrderedDict()
        self.ActiveOptions = OrderedDict()
        self.ForcedOptions = OrderedDict()
        self.ForcedWarnings = OrderedDict()
        self.InactiveOptions = OrderedDict()
        self.InactiveWarnings = OrderedDict()
        # First build a dictionary of user supplied options.
        if input_file != None:
            for line in open(input_file).readlines():
                line = sub('#.*$','',line.strip())
                s = line.split()
                if len(s) > 0:
                    # Options are case insensitive
                    key = s[0].lower()
                    try:
                        val = leval(line.replace(s[0],'',1).strip())
                    except:
                        val = str(line.replace(s[0],'',1).strip())
                    self.UserOptions[key] = val



    def set_active(self,key,default,typ,doc,allowed=None,depend=True,clash=False,msg=None):
        """ Set one option.  The arguments are:
        key     : The name of the option.
        default : The default value.
        typ     : The type of the value.
        doc     : The documentation string.
        allowed : An optional list of allowed values.
        depend  : A condition that must be True for the option to be activated.
        clash   : A condition that must be False for the option to be activated.
        msg     : A warning that is printed out if the option is not activated.
        """
        doc = sub("\.$","",doc.strip())+"."
        self.Documentation[key] = "%-8s " % ("(" + sub("'>","",sub("<type '","",str(typ)))+")") + doc
        if key in self.UserOptions:
            val = self.UserOptions[key]
        else:
            val = default
        if type(allowed) is list:
            self.Documentation[key] += " Allowed values are %s" % str(allowed)
            if val not in allowed:
                raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not allowed (choose from \x1b[92m%s\x1b[0m)" % (key, str(val), str(allowed)))
        if typ is bool and type(val) == int:
            val = bool(val)
        if val != None and type(val) is not typ:
            raise Exception("Tried to set option \x1b[1;91m%s\x1b[0m to \x1b[94m%s\x1b[0m but it's not the right type (%s required)" % (key, str(val), str(typ)))
        if depend and not clash:
            if key in self.InactiveOptions:
                del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
        else:
            if key in self.ActiveOptions:
                del self.ActiveOptions[key]
            self.InactiveOptions[key] = val
            self.InactiveWarnings[key] = msg

    def force_active(self,key,val=None,msg=None):
        """ Force an option to be active and set it to the provided value,
        regardless of the user input.  There are no safeguards, so use carefully.

        key     : The name of the option.
        val     : The value that the option is being set to.
        msg     : A warning that is printed out if the option is not activated.
        """
        print(key)
        print(val)
        if msg == None:
            msg == "Option forced to active for no given reason."
        if key not in self.ActiveOptions:
            if val == None:
                val = self.InactiveOptions[key]
            del self.InactiveOptions[key]
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val != None and self.ActiveOptions[key] != val:
            self.ActiveOptions[key] = val
            self.ForcedOptions[key] = val
            self.ForcedWarnings[key] = msg
        elif val == None:
            self.ForcedOptions[key] = self.ActiveOptions[key]
            self.ForcedWarnings[key] = msg + " (Warning: Forced active but it was already active.)"

    def deactivate(self,key,msg=None):
        """ Deactivate one option.  The arguments are:
        key     : The name of the option.
        msg     : A warning that is printed out if the option is not activated.
        """
        if key in self.ActiveOptions:
            self.InactiveOptions[key] = self.ActiveOptions[key]
            del self.ActiveOptions[key]
        self.InactiveWarnings[key] = msg

    def __getattr__(self,key):
        if key in self.ActiveOptions:
            return self.ActiveOptions[key]
        elif key in self.InactiveOptions:
            return None
        else:
            return getattr(super(File_Options,self),key)

    def record(self):
        out = []
        TopBar = False
        UserSupplied = []
        for key in self.ActiveOptions:
            if key in self.UserOptions and key not in self.ForcedOptions:
                UserSupplied.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(UserSupplied) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|          User-supplied options:         |#")
            out.append("#===========================================#")
            out += UserSupplied
        Forced = []
        for key in self.ActiveOptions:
            if key in self.ForcedOptions:
                Forced.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
                Forced.append("%-22s %20s # Reason : %s" % ("","",self.ForcedWarnings[key]))
        if len(Forced) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|     Options enforced by the script:     |#")
            out.append("#===========================================#")
            out += Forced
        ActiveDefault = []
        for key in self.ActiveOptions:
            if key not in self.UserOptions and key not in self.ForcedOptions:
                ActiveDefault.append("%-22s %20s # %s" % (key, str(self.ActiveOptions[key]), self.Documentation[key]))
        if len(ActiveDefault) > 0:
            if TopBar:
                out.append("#===========================================#")
            else:
                TopBar = True
            out.append("#|   Active options at default values:     |#")
            out.append("#===========================================#")
            out += ActiveDefault
        # out.append("")
        out.append("#===========================================#")
        out.append("#|           End of Input File             |#")
        out.append("#===========================================#")
        Deactivated = []
        for key in self.InactiveOptions:
            Deactivated.append("%-22s %20s # %s" % (key, str(self.InactiveOptions[key]), self.Documentation[key]))
            Deactivated.append("%-22s %20s # Reason : %s" % ("","",self.InactiveWarnings[key]))
        if len(Deactivated) > 0:
            out.append("")
            out.append("#===========================================#")
            out.append("#|   Deactivated or conflicting options:   |#")
            out.append("#===========================================#")
            out += Deactivated
        Unrecognized = []
        for key in self.UserOptions:
            if key not in self.ActiveOptions and key not in self.InactiveOptions:
                Unrecognized.append("%-22s %20s" % (key, self.UserOptions[key]))
        if len(Unrecognized) > 0:
            # out.append("")
            out.append("#===========================================#")
            out.append("#|          Unrecognized options:          |#")
            out.append("#===========================================#")
            out += Unrecognized
        return out

