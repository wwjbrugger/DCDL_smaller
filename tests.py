import numpy as np
import SLS_Algorithm
import helper_methods  as help
import model.boolean_Formel as bofo
import own_scripts.ripper_by_wittgenstein as ripper
import pickle


def test_one_class_against_all():
    one_hot_vector = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    index_one_against_all_target = \
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    index_five_against_all_target = \
        np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    converted_one_hot_vector = help.one_class_against_all(one_hot_vector, one_class=1, number_classes_output=10)
    assert np.array_equal(index_one_against_all_target, converted_one_hot_vector)

    converted_one_hot_vector = help.one_class_against_all(one_hot_vector, one_class=5, number_classes_output=10)
    assert np.array_equal(index_five_against_all_target, converted_one_hot_vector)


def test_boolsche_formel_1():
    formel_1 = bofo.Boolsche_formel(np.array([255, 4, 24, 16], dtype=np.uint8),
                                    np.array([4, 4, 24, 16], dtype=np.uint8), number_of_product_term=2)

    target_output_relvant = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
                                      np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])
    target_output_negated = np.array([np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
                                      np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])

    target_output_formula_in_arrays_code = np.array(
        [np.array([-1, -1, -1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int64),
         np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64)])

    assert np.array_equal(target_output_relvant, formel_1.pixel_relevant_in_arrays_code)
    assert np.array_equal(target_output_negated, formel_1.pixel_negated_in_arrays_code)
    assert np.array_equal(target_output_formula_in_arrays_code, formel_1.formel_in_arrays_code)
    formel_1.pretty_print_formula()
    formel_1.built_plot(0, 'Boolsche Formel from Test 1')


def test_split_formula():
    input = np.array([-1, -1, -1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0])
    target_output_relevant = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    target_output_negated = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    output_relevant, output_negated = bofo.Boolsche_formel.split_fomula(input)
    assert np.array_equal(target_output_relevant, output_relevant)
    assert np.array_equal(target_output_negated, output_negated)


def test_boolsche_formel_1_Belegung():
    formel_1 = bofo.Boolsche_formel(np.array([255, 4, 24, 16], dtype=np.uint8),
                                    np.array([4, 4, 24, 16], dtype=np.uint8), number_of_product_term=2)

    Belegung = np.array([np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # True
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # True
                         np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool),  # False
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool),  # False
                         np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)])  # False

    target_output = np.array([False, False, True, False, False])

    assert np.array_equal(formel_1.evaluate_belegung_like_c(Belegung), target_output)


def test_boolsche_formel_2():
    formel_2 = bofo.Boolsche_formel(np.array([255, 4], dtype=np.uint8), np.array([24, 16], dtype=np.uint8),
                                    number_of_product_term=1)
    target_output_relvant = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8)])
    target_output_negated = np.array([np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)])
    target_output_formula = np.array(
        [np.array([-1, -1, -1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0], dtype=np.int64)])

    assert np.array_equal(target_output_relvant, formel_2.pixel_relevant_in_arrays_code)
    assert np.array_equal(target_output_negated, formel_2.pixel_negated_in_arrays_code)
    assert np.array_equal(target_output_formula, formel_2.formel_in_arrays_code)
    formel_2.pretty_print_formula()
    formel_2.built_plot(0, 'Boolsche Formel from Test 2 \n ')


def creat_matrix_with_1_at_a_index(index_variable_to_fill_with_1, shape=(8, 8)):
    data = []
    label = []

    for row in range(shape[0]):
        row_data = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        if row % 2 == 0:
            row_data[index_variable_to_fill_with_1] = 1
            data.append(row_data)
            label.append(1)
        else:
            data.append(row_data)
            label.append(0)
    return np.array(data), np.array(label)


def test_transform_single_number_code_in_arrays_code():
    Boolsche_formel_object = bofo.Boolsche_formel(np.array([255], dtype=np.uint8),
                                                  # Need a object of Boolsche_formel to test methods of this class
                                                  np.array([24], dtype=np.uint8),
                                                  number_of_product_term=1)
    input_single_1 = np.array([200], dtype=np.uint8)
    input_single_2 = np.array([255], dtype=np.uint8)
    input_single_3 = np.array([0], dtype=np.uint8)
    target_output_1 = np.array([np.array([1, 1, 0, 0, 1, 0, 0, 0])])
    target_output_2 = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1])])
    target_output_3 = np.array([np.array([0, 0, 0, 0, 0, 0, 0, 0])])
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_1), target_output_1)
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_2), target_output_2)
    assert np.array_equal(Boolsche_formel_object.transform_number_code_in_arrays_code(input_single_3), target_output_3)


def test_transform_multi_number_code_in_multi_arrays_code_1():
    boolsche_formel_object_number_of_product_term_1 = bofo.Boolsche_formel(np.array([255, 4], dtype=np.uint8),
                                                                           # Need a object of Boolsche_formel to test methods of this class
                                                                           np.array([24, 16], dtype=np.uint8),
                                                                           number_of_product_term=2)
    input_multi_1 = np.array([200, 255, 0], dtype=np.uint8)

    target_multi_1 = np.array(
        [np.array([1, 1, 0, 0, 1, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 0, 0, 0])])

    assert np.array_equal(boolsche_formel_object_number_of_product_term_1.
                          transform_number_code_in_arrays_code(input_multi_1), target_multi_1)


def test_transform_multi_number_code_in_arrays_code():
    boolsche_formel_object_number_of_product_term = bofo.Boolsche_formel(np.array([255, 4, 5, 6], dtype=np.uint8),
                                                                         np.array([24, 16, 6, 6], dtype=np.uint8),
                                                                         number_of_product_term=2)  # Need a object of Boolsche_formel to test methods of this class
    input_multi = np.array([200, 255, 0, 200], dtype=np.uint8)
    target_multi = np.array(
        [np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, ]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])])
    assert np.array_equal(boolsche_formel_object_number_of_product_term.
                          transform_number_code_in_arrays_code(input_multi), target_multi)


def test_trasform_arrays_code_in_numbercode():
    array_code = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    target = np.array([200, 255])
    result = bofo.Boolsche_formel.transform_arrays_code_in_number_code(array_code)
    assert np.array_equal(result, target)

    array_code = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    target = np.array([200, 254])
    result = bofo.Boolsche_formel.transform_arrays_code_in_number_code(array_code)
    assert np.array_equal(result, target)


def test_sls_algorithm_easiest():
    index_variable_to_fill_with_1 = 6
    data, label = creat_matrix_with_1_at_a_index(index_variable_to_fill_with_1, shape=(8, 8))

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 1000)

    assert np.array_equal(label, found_formula.evaluate_belegung_like_c(data))
    found_formula.pretty_print_formula("graphische Repräsentation der extrahierten Regeln für einfachsten Datensatz")
    found_formula.built_plot(0, "graphische Repräsentation der extrahierten Regeln für einfachsten Datensatz")


def test_sls_algorithm_1():
    data = np.array(
        [np.array([1, 1, 1, 0, 0, 0, 0, 1]),
         np.array([1, 0, 0, 0, 0, 0, 0, 1]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1])])
    label = np.array([1,
                      0,
                      1,
                      0])

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 2, 100)
    print('Pixel_relevant_in_number_code: ', found_formula.pixel_relevant_in_number_code)
    print('Pixel_relevant_in_arrays_code: ', found_formula.pixel_relevant_in_arrays_code)
    print('pixel_negated_in_arrays_code: ', found_formula.pixel_negated_in_arrays_code)
    assert np.equal(help.prediction_SLS_fast(data, label, found_formula), 1)
    result = np.array(found_formula.evaluate_belegung_like_c(data))
    print('evaluate_belegung_python', result)
    assert np.array_equal(result, label)

    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')


def test_sls_algorithm_2():  # (x_0 and x_2)
    data = np.array(
        [np.array([1, 1, 1, 0, 1, 1, 1, 1]),
         np.array([0, 1, 1, 0, 1, 1, 1, 1]),
         np.array([0, 1, 1, 0, 1, 1, 1, 1]),
         np.array([1, 1, 1, 1, 1, 1, 1, 1])])
    label = np.array([1,
                      0,
                      0,
                      0])
    data = help.transform_to_boolean(data)
    label = help.transform_to_boolean(label)
    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 10000)
    result = np.array(found_formula.evaluate_belegung_like_c(data))
    print('evaluate_belegung', result)
    np.array_equal(label, result)
    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')
    found_formula.built_plot(0, 'Formula for test_sls_algorithm_2')


def gen_data():
    data = np.array(
        [np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),  # True
         np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0]),  # True
         np.array([1, 1, 1, 0, 1, 1, 1, 1, 0]),  # True
         np.array([0, 1, 1, 0, 1, 0, 1, 1, 0]),  # False
         np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),  # False
         np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]),  # True
         np.array([1, 0, 0, 0, 0, 1, 0, 0, 0]),  # True
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # False
         np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]),  # True
         np.array([0, 0, 1, 0, 1, 1, 0, 0, 0])  # True
         ])
    label = np.array([0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1,
                      0,
                      0,
                      0,
                      0,
                      1,
                      0,
                      1,
                      0,
                      1,
                      1
                      ])

    return data, label


def test_sls_algorithm_3():  # (x_0 and x_2) or x_5
    data, label = gen_data()
    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 2, 10000, )
    found_formula.number_of_relevant_variabels = 9

    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_3()')
    assert np.array_equal(label, found_formula.evaluate_belegung_like_c(data))
    found_formula.built_plot(0, 'conjunction 0 ')
    found_formula.built_plot(1, 'conjunction 1 ')


def test_transform_to_boolean():
    input = [[1, -1, 0]]
    target_output = [[1, 0, 0]]
    assert np.array_equal(help.transform_to_boolean(input), target_output)


def test_data_in_kernel():
    input = np.array([[  # picture
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]])


    target_output = np.array(
        [[[[0], [1]],
          [[4], [5]]],

         [[[1], [2]],
          [[5], [6]]],

         [[[2], [3]],
          [[6], [7]]],

         [[[3], [0]],
          [[7], [0]]],

         [[[4], [5]],
          [[8], [9]]],

         [[[5], [6]],
          [[9], [10]]],

         [[[6], [7]],
          [[10], [11]]],

         [[[7], [0]],
          [[11], [0]]],

         [[[8], [9]],
          [[12], [13]]],

         [[[9], [10]],
          [[13], [14]]],

         [[[10], [11]],
          [[14], [15]]],

         [[[11], [0]],
          [[15], [0]]],

         [[[12], [13]],
          [[0], [0]]],

         [[[13], [14]],
          [[0], [0]]],

         [[[14], [15]],
          [[0], [0]]],

         [[[15], [0]],
          [[0], [0]]]])

    result = help.data_in_kernel(input, stepsize=1, width=2)
    assert np.array_equal(target_output, result)

    target_output_2 = np.array([[[[0], [1]],
                                 [[4], [5]]],

                                [[[2], [3]],
                                 [[6], [7]]],

                                [[[8], [9]],
                                 [[12], [13]]],

                                [[[10], [11]],
                                 [[14], [15]]]])
    result = help.data_in_kernel(input, stepsize=2, width=2)
    assert np.array_equal(target_output_2, result)


def test_ripper_by_wittgenstein():
    data, label = gen_data()

    df = ripper.np_to_padas(data, label)
    rule_set, accuracy = ripper.wittgenstein_ripper(df, 'label', max_rules=2)
    print('rule set: \n', rule_set, '\n accuracy:', accuracy)

    found_formula = SLS_Algorithm.rule_extraction_with_sls_without_validation(data, label, 1, 10000, )
    found_formula.number_of_relevant_variabels = 9

    found_formula.pretty_print_formula('DNF found for test_sls_algorithm_1()')


def test_extraction_with_real_data():
    Number_of_Product_term = 2
    Maximum_Steps_in_SKS = 10000

    training_set = np.load('data/real_data_for_extraction/data_for_SLS.npy')
    label_set = np.load('data/real_data_for_extraction/label_SLS.npy')
    kernel = np.load('data/real_data_for_extraction/kernel.npy')
    # bias = np.load('data/real_data_for_extraction/bias.npy')
    result_conv = np.load('data/real_data_for_extraction/result_conv.npy')
    kernel_width = kernel.shape[0]

    training_set_kernel_int = help.data_in_kernel(training_set, stepsize=2, width=kernel_width)
    training_set_kernel_int = training_set_kernel_int.reshape((196, -1))
    # training_set_kernel = help.transform_to_boolean(training_set_kernel_int)
    sign_kernel = np.sign(kernel)
    result = np.zeros(label_set.shape)
    for channel in range(1):  # label_set.shape[3]):
        label_self_calculated = help.calculate_convolution(training_set_kernel_int, kernel[:, :, :, channel],
                                                           result_conv)
        result[0, :, :, channel] = np.reshape(label_self_calculated, (14,14))
    label_self_sign = np.sign(result)
    np.array_equal(result, label_set)


def test_reduce_kernel():
    input = [[[1, 1],
              [1, 1]],

             [[2, -1],
              [3, -1]],

             [[3, 0],
              [-10, -2]]]

    target_sum = [[6, 0],
                   [-6, -2]]

    target_mean = [[2, 0],
                    [-2, -2 / 3]]

    target_min_max = [[1, 0],
                       [-1, -2 / 6]]

    target_norm = [[13 / 13, 1 / 13],
                    [-11 / 13, -3 / 13]]

    assert np.array_equal(help.reduce_kernel(input, mode='sum'), target_sum)
    assert np.array_equal(help.reduce_kernel(input, mode='mean'), target_mean)
    assert np.array_equal(help.reduce_kernel(input, mode='min_max'), target_min_max)
    assert np.array_equal(help.reduce_kernel(input, mode='norm'), target_norm)


def test_permutate_and_flaten_single_channel():
    data = np.array([[[[1], [2]],  # pic,
                      [[3], [4]]],
                     [[[-1], [-2]],  # pic,
                      [[-3], [-4]]]])

    data_target = np.array([[1, 2, 3, 4], [-1, -2, -3, -4]])

    label = np.array([[[[1]]], [[[3]]]])

    label_target = np.array([1, 3])
    training_set_flat, label_set_flat = help.permutate_and_flaten(data, label,  channel_label=0)

    np.array_equal(data_target, training_set_flat)
    np.array_equal(label_target, label_set_flat)


def test_permutate_and_flaten_input_multi_channel ():
    data = np.array([[[[1, 5], [2, 6]],  # pic,
                      [[3, 7], [4, 8]]],
                     [[[-1, -5], [-2, -6]],  # pic,
                      [[-3, -7], [-4, -8]]]
                     ])
    data_target = np.array([[1,2,3,4,5,6,7,8], [-1,-2,-3,-4,-5,-6,-7,-8]])

    label = np.array([[[[1, 2]]], [[[3, 4]]]])
    label_target =  np.array([1, 3])

    training_set_flat, label_set_flat = help.permutate_and_flaten(data, label, channel_label=0)
    np.array_equal(data_target, training_set_flat)
    np.array_equal(label_target, label_set_flat)

    training_set_flat, label_set_flat = help.permutate_and_flaten(data, label, channel_label=1)
    label_target = np.array([2, 4])
    np.array_equal(data_target, training_set_flat)
    np.array_equal(label_target, label_set_flat)




def test_reshape():
    data = np.array(
        [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]], [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]]])

    shape = (2, 2, 2, 3)
    data_flat = np.reshape(data, -1)
    assert np.array_equal(data, np.reshape(data_flat, shape))

def test_dither_pic():
    import data.mnist_fashion as fashion
    import data.cifar_dataset as cifar
    import matplotlib.pyplot as plt
    import dithering_diffusion as dith
    import os

    for dataset in [cifar.data(), fashion.data()]:
        dataset.get_iterator()
        train_nn, label_train_nn = dataset.get_chunk(30)
        if train_nn.ndim == 3:
            class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            train_nn = train_nn.reshape((train_nn.shape + (1,)))
        else:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        file_name_origin = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data',
                                '{}_dither'.format(dataset.name_dataset()), '{}.png'.format('origin'))
        help.visualize_pic(train_nn, label_train_nn, class_names,
                           " pic {}".format('origin'), plt.cm.Greys, filename=file_name_origin)
        for method in [ 'Floyd-Steinberg', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes', 'sierra3',  'sierra2', 'sierra-2-4a', 'stevenson-arce']: # 270 pic
            file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', '{}_dither'.format(dataset.name_dataset()), '{}.png'.format(method))
            train_nn_dither = dith.error_diffusion_dithering(train_nn, method=method)
            help.visualize_pic(train_nn_dither, label_train_nn, class_names,
                               " pic after dithering Method = {}".format(method), plt.cm.Greys, filename=file_name)

        #10 Bilder = 3 sec



