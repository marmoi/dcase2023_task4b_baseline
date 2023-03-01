device = 'cuda'
posterior_thresh = 0.5
sample_rate = 44100
hop_size = 8820

# 17 classes
labels_soft = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 'wind_blowing',
          'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 'furniture dragging', 'coffee machine',
          'metro approaching', 'metro leaving', 'door opens/closes', 'announcement', 'shopping cart',
          'cash register beeping']

class_labels_soft = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'furniture dragging': 9,
    'coffee machine': 10,
    'metro approaching': 11,
    'metro leaving': 12,
    'door opens/closes': 13,
    'announcement': 14,
    'shopping cart': 15,
    'cash register beeping': 16
}

classes_num_soft = len(labels_soft)



# For the hard labels we have 11 classes
labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 'children voices', 'wind_blowing',
          'brakes_squeaking', 'large_vehicle', 'cutlery and dishes', 'metro approaching', 'metro leaving']

class_labels_hard = {
    'birds_singing': 0,
    'car': 1,
    'people talking': 2,
    'footsteps': 3,
    'children voices': 4,
    'wind_blowing': 5,
    'brakes_squeaking': 6,
    'large_vehicle': 7,
    'cutlery and dishes': 8,
    'metro approaching': 9,
    'metro leaving': 10,
    }

classes_num_hard = len(labels_hard)