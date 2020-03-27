def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride
    window_nums = []

    for _, ratio in enumerate(ratios):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))

    return window_nums