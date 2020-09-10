function getNodes(str) {
    return new DOMParser().parseFromString(str, 'text/html').body.children;
}

function getNode(str) {
    return getNodes(str)[0];
}


const request = (method, url, payload, rawPayload = false) => {
    return new Promise(function (resolve, reject) {
        let xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function () {
            if (this.readyState === 4) {
                if (this.status === 200) {
                    let r = this.responseText;
                    if (this.getResponseHeader("Content-Type") === 'application/json')
                        r = JSON.parse(r);
                    resolve(r);
                } else {
                    reject('FAIL');
                }
            }
        };
        xhttp.open(method, url, true);
        xhttp.setRequestHeader('Accept', 'application/json');
        xhttp.setRequestHeader('Accept', 'text/plain');
        xhttp.send(rawPayload ? payload : JSON.stringify(payload));
    });
};

const GET = (url) =>
    request('GET', url);

const POST = (url, payload, rawPayload = false) =>
    request('POST', url, payload, rawPayload);

class DataHandler {
    #staticDatasetPath = '../data/static/';
    #datasetListPath = '../data/list';

    constructor(datasetSelectorElem, onChangeCallback) {
        this.datasetSelectorElem = datasetSelectorElem;
        this.onChangeCallback = onChangeCallback;
        this.data = [];

        return new Promise(async (resolve, reject) => {
            this.#populateDropdown()
                .then(() => {
                    this.datasetSelectorElem.addEventListener('change', this.onChange.bind(this));
                    resolve(this);
                })
                .catch((reason => reject(reason)));
        });
    }

    async #populateDropdown() {
        const datasets = await GET(this.#datasetListPath);
        console.log(datasets)

        datasets.forEach((dataset) => {
            const node = getNode(`<option value="${dataset}">${dataset}</option>`);
            this.datasetSelectorElem.appendChild(node);
        });
    }

    getSelectedDataset() {
        return this.datasetSelectorElem.options[this.datasetSelectorElem.selectedIndex].value;
    }

    async #requestDataset(dataset) {
        return await GET(this.#staticDatasetPath + dataset);
    }

    parseDataset(responseText) {
        return responseText.split('\n').map(line => {
            const parts = line.split(' ');
            return [
                parseFloat(parts[0]) / 22, // x
                parseFloat(parts[1]) / 22, // y
                parseInt(parts[2]), // category
                Math.random(), // value
            ]
        });
    }

    async onChange() {
        this.data = this.parseDataset(await this.#requestDataset(this.getSelectedDataset()));
        this.onChangeCallback(this.data);
    }
}

export { DataHandler };