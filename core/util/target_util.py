from .attr_config import attribution_config

def get_attr_id_from_json(attributions):
    key_list = get_attr_value(attributions)
    key = list()
    for index, value in attribution_config.items():
        if value['attrInfo'] in key_list:
            key.append(index)
    return key

def get_attr_value(attributions):
    key_list = []
    for key, val in attributions.items():
        if val is None:
            continue
        for attr_key, attr_val in val.items():
            if attr_val in [None, '']:
                continue
            elif attr_val:
                if attr_key == 'color':
                    for value in attr_val:
                        if value not in [None, '']:
                            attr_info = f"{key}_{value}"
                            key_list.append(attr_info)
                else:
                    attr_info = f"{key}_{attr_val}"
                    key_list.append(attr_info)
    return key_list