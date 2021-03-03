from torch_receptive_field import receptive_field, receptive_field_for_unit

def get_receptive_field_dict(net, input_shape=(1,128,128)):
    receptive_field_dict = None
    receptive_field_dict = receptive_field(net, input_shape)
    return receptive_field_dict

def get_receptive_field_for_unit(net, input_shape, layer_number, h, w):
    receptive_field_dict = get_receptive_field_dict(net, input_shape)
    return receptive_field_for_unit(receptive_field_dict, str(layer_number), (h,w))

def get_receptive_field_for_units(net,input_shape,layer_number, h, w):
    receptive_field_dict = get_receptive_field_dict(net, input_shape)
    units_to_fields = {}
    for x in range(h):
        for y in range(w):
            units_to_fields[(x,y)] = receptive_field_for_unit(receptive_field_dict, str(layer_number), (x,y))
    return units_to_fields
