labels = open('Viewer/js/regl-scatterplot/data/mnist_label.txt', 'r')
coords = open('Viewer/js/regl-scatterplot/data/mnist_LargeVis_.txt', 'r')
out = open('Viewer/js/regl-scatterplot/data/mnist_LargeVis.txt', 'w')
for label, coord in zip(labels, coords):
    out.write(coord.strip() + ' ' + label)
