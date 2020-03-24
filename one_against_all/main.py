import one_against_all.one_against_all_train as first
import one_against_all.one_against_all_data_generation as secound
import one_against_all.extracting_pictures as third
import one_against_all.reduce_kernel as fourths

if __name__ == '__main__':
    number_classes_to_predict = 2


    dithering_used= True
    one_against_all = 7

    Number_of_disjuntion_term_in_SLS = 2000
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 28


    network = first.model_one_convolution.network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 1000, stride=stride_of_convolution, check_every= 200, number_of_kernel=1,
                                                            number_classes=number_classes_to_predict)


    first.prepare_dataset_and_train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)

    secound.one_against_all_data_generation(network)

    third.visualize_kernel(one_against_all)

    for i in range(3):
        third.sls_on_data_of_the_neural_network(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, one_against_all)
        fourths.reduce_SLS_results()