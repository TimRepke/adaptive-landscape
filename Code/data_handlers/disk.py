from string import Template
import os
import logging

logger = logging.getLogger('disk')


def store_result(vectors, labels, dataset, algorithm, interval, target_folder, parameters=None,
                 base_template='${dataset}/${algorithm}_${interval}.tsv'):
    if parameters is None:
        parameters = {}
    for parameter in parameters.keys():
        base_template += '_' + parameter + '${' + parameter + '}'
    parameters['dataset'] = dataset
    parameters['algorithm'] = algorithm
    parameters['interval'] = interval
    file_name = os.path.join(target_folder, Template(base_template).substitute(parameters))
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    logger.info(f'> Writing output to {file_name}...')
    with open(file_name, 'w') as out:
        [
            out.write(' '.join(f'{dim:.5f}' for dim in vector) + f' {label}\n')
            for vector, label in zip(vectors, labels)
        ]
