import numpy as np

## too much negative bias
def revert_elements_2d_mid(arr):
    flat_arr = arr.flatten()
    
    # Calculate the number of elements to revert (80% of the total elements)
    num_elements_to_revert = int(len(flat_arr) * 0.8)
    print(len(flat_arr))
    print(num_elements_to_revert)
    
    # Find the indices of the elements to revert
    indices_to_revert = np.argsort(flat_arr)[:num_elements_to_revert]
    
    for idx in indices_to_revert:
        flat_arr[idx] = 0
    
    reverted_arr = flat_arr.reshape(arr.shape)
    
    return reverted_arr

## too much negative bias
def revert_elements_2d_edges(arr):
    flat_arr = arr.flatten()
    
    # Calculate the number of bins to revert (80% of the total bins)
    unique_elements, counts = np.unique(flat_arr, return_counts=True)
    num_bins_to_revert = int(len(unique_elements) * 0.8)
    
    # Calculate the number of bins to revert from each side (40% from the left and 40% from the right)
    num_bins_each_side = num_bins_to_revert // 2
    
    # Find the bins to revert from the left and right edges
    sorted_unique_elements = np.sort(unique_elements)
    print(sorted_unique_elements)
    bins_to_revert = np.concatenate((sorted_unique_elements[:num_bins_each_side], sorted_unique_elements[-num_bins_each_side:]))
    print(bins_to_revert)
    
    for bin_value in bins_to_revert:
        flat_arr[flat_arr == bin_value] = 0
    
    reverted_arr = flat_arr.reshape(arr.shape)
    
    return reverted_arr


def revert_elements_2d_mid_separate(arr):
    flat_arr = arr.flatten()
    
    # Separate negative and positive elements (except 0)
    negative_elements = flat_arr[flat_arr < 0]
    positive_elements = flat_arr[flat_arr > 0]
    
    # Calculate the number of elements to revert for negative and positive elements (except 0)
    num_elements_to_revert_negative = int(len(negative_elements) * 0.8)
    num_elements_to_revert_positive = int(len(positive_elements) * 0.8)

    # Find the indices of the elements to revert
    indices_to_revert_negative = np.argsort(negative_elements)[:num_elements_to_revert_negative]
    indices_to_revert_positive = np.argsort(positive_elements)[:num_elements_to_revert_positive]

    for idx in indices_to_revert_negative:
        flat_arr[np.where(flat_arr == negative_elements[idx])[0][0]] = 0
    for idx in indices_to_revert_positive:
        flat_arr[np.where(flat_arr == positive_elements[idx])[0][0]] = 0
    
    reverted_arr = flat_arr.reshape(arr.shape)
    
    return reverted_arr


def revert_elements_2d_edges_separate(arr):
    flat_arr = arr.flatten()
    
    # Separate negative and positive elements (except 0)
    negative_elements = flat_arr[flat_arr < 0]
    positive_elements = flat_arr[flat_arr > 0]
    
    # Calculate the number of bins (unique elements) for negative and positive elements
    unique_negative_elements, counts_negative = np.unique(negative_elements, return_counts=True)
    unique_positive_elements, counts_positive = np.unique(positive_elements, return_counts=True)
    
    # Calculate the number of bins to revert (80% of the total bins for each side)
    num_bins_to_revert_negative = int(len(unique_negative_elements) * 0.8)
    num_bins_to_revert_positive = int(len(unique_positive_elements) * 0.8)
    
    # Sort the unique elements to find the edges of the binomial curve
    sorted_unique_negative_elements = np.sort(unique_negative_elements)
    sorted_unique_positive_elements = np.sort(unique_positive_elements)
    
    # Find the bins to revert from the left and right edges
    bins_to_revert_negative = sorted_unique_negative_elements[:num_bins_to_revert_negative]
    bins_to_revert_positive = sorted_unique_positive_elements[-num_bins_to_revert_positive:]
    
    for bin_value in bins_to_revert_negative:
        flat_arr[flat_arr == bin_value] = 0
    for bin_value in bins_to_revert_positive:
        flat_arr[flat_arr == bin_value] = 0
    
    
    reverted_arr = flat_arr.reshape(arr.shape)
    
    return reverted_arr


if __name__ == '__main__':
    file_path = "ind_off/ind_off_4_init.txt"
    with open(file_path, 'r') as file:
        arr = np.loadtxt(file)

    # arr = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5], [3, 5, 8]])
    print(arr)
    print("")
    # reverted_arr = revert_elements_2d_mid(arr)
    reverted_arr = revert_elements_2d_mid_separate(arr)
    print(reverted_arr)


    print("")
    print("-----------------------------------")
    print("")


    # # arr = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5], [3, 5, 8]])
    # print(arr)
    # print("")
    # # reverted_arr = revert_elements_2d_edges(arr)
    # reverted_arr = revert_elements_2d_edges_separate(arr)
    # print(reverted_arr)

