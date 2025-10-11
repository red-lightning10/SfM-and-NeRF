import numpy as np

def create_visibility_dict(n):
    visibility = {}
    for i in range(n):
        visibility[i + 1] = {}
    return visibility

def filter_visibility_dict(visibility, filtered_dict, filtered_matches, i, j, num_images = 5):
    for match in filtered_matches:
        feature_x = match[0][0]
        feature_y = match[0][1]
        filtered_dict[i+1][str([feature_x, feature_y])] = visibility[i+1][str([feature_x, feature_y])]
        feature_x = match[1][0]
        feature_y = match[1][1]
        filtered_dict[j+1][str([feature_x, feature_y])] = visibility[j+1][str([feature_x, feature_y])]
    return filtered_dict

def access_visibility_dictionary(visibility_dictionary, feature_x, feature_y, i):
    key = str([feature_x, feature_y])
    if key in visibility_dictionary[i + 1]:
        return visibility_dictionary[i + 1][key]
    else:
        return np.zeros(5, dtype=bool)

def get_features_and_visibility(visibility_dictionary, feature_points, i):
    features_x = []
    features_y = []
    visibility = []
    for point in feature_points:
        feature_x = point[0]
        feature_y = point[1]
        visibility.append(access_visibility_dictionary(visibility_dictionary, feature_x, feature_y, i).flatten())
        features_x.append(feature_x)
        features_y.append(feature_y)
    return features_x, features_y, visibility