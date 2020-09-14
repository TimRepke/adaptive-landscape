from string import Template


def store_result(vectors, labels, dataset, algorithm, interval, parameters=None,
                 base_template='$dataset_$algorithm_$interval'):
    if parameters is None:
        parameters = {}
    for parameter in parameters.keys():
        base_template += '_' + parameter + '$' + parameter
    parameters['dataset'] = dataset
    parameters['algorithm'] = algorithm
    parameters['interval'] = interval
    with open(Template(base_template).substitute(parameters), 'w') as out:
        for vector, label in zip(vectors, labels):
            out.write(' '.join(f'{dim:.5f}' for dim in vector) + f' {label}\n')
