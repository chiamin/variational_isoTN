import re, copy

DUMMY_NUMBER = 0

def CheckDuplicate (network_dict):
    for name, inds in network_dict.items():
        tmp = set(inds)
        if len(tmp) != len(inds):
            print("Duplicate index in a tensor")
            print(name,":",inds)
            raise Exception

def GetOpenIndices (network_dict):
    # Store all the indices
    inds_all = []
    for name, inds in network_dict.items():
        inds_all.extend(inds)

    # Store the open indices which appear only once
    open_inds = []
    for ind in inds_all:
        if inds_all.count(ind) == 1:
            open_inds.append(ind)
    return open_inds

def ToNetworkString (network_dict):
    CheckDuplicate (network_dict)

    # Get all the open indices
    if "TOUT" not in network_dict:
        open_inds = GetOpenIndices(network_dict)
    else:
        open_inds = network_dict["TOUT"]

    # Store the dummy indices
    dummy_inds = set()
    for name in network_dict:
        # Store the dummy indices
        for ind in network_dict[name]:
            if ind not in open_inds:
                dummy_inds.add(ind)

    # For each dummy index, create a dummy name
    dummy_names = dict()
    for ind in dummy_inds:
        global DUMMY_NUMBER
        DUMMY_NUMBER += 1
        dummy_names[ind] = str(DUMMY_NUMBER)

    # Get the string
    res = []
    for name, inds in network_dict.items():
        if name != "TOUT":
            inds_new = []
            for ind in inds:
                # If dummy index, use the dummy name
                if ind not in open_inds:
                    ind = dummy_names[ind]
                inds_new.append(ind)
            ind_str = ", ".join(inds_new)
            res.append (name+": "+ind_str)
    # For TOUT
    ind_str = ", ".join(open_inds)
    res.append("TOUT: "+ind_str)
    return res

def Prime (network_dict):
    new_dict = dict()
    for name in network_dict:
        if name != "TOUT":
            name_new = name+"'"
        else:
            name_new = name

        inds_new = [ind+"'" for ind in network_dict[name]]
        new_dict[name_new] = inds_new
    return new_dict

def ReplaceIndex (network_dict, old_inds, new_inds):
    if type(old_inds) == str:
        assert type(new_inds) == str
        old_inds = [old_inds]
        new_inds = [new_inds]
    assert len(old_inds) == len(new_inds)

    # Make a dictionary for the replacing indices
    replace_table = dict(zip(old_inds, new_inds))

    new_dict = dict()
    for name, inds in network_dict.items():
        new_inds = []
        for ind in inds:
            if ind in replace_table:
                new_inds.append (replace_table[ind])
            else:
                new_inds.append (ind)
        new_dict[name] = new_inds
    return new_dict

def Combine (network_dict1, network_dict2):
    new_dict = network_dict1 | network_dict2
    if "TOUT" in new_dict:
        del new_dict["TOUT"]
    return new_dict

# Count the order of tensor <name> in a network <pattern>
def CountOrder (pattern, name):
    count = 0
    for p in pattern:
        c = p.rstrip("'")
        if c == name:
            count += 1
    return count

