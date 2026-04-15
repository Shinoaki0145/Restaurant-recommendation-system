def merge_dict(*args:dict):
    result = {}
    if len(args) > 0:
        for keys in args[0].keys():
            result.update({keys: []})
        
        for item in args:
            for key in item.keys():
                result[key].append(item[key])          
    return result 
