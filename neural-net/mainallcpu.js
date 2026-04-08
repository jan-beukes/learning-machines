"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { Worker, isMainThread, parentPort } = require("worker_threads");

const MNIST_RES = 28;
const CIFAR_RES = 32;
const CIFAR_BATCH_IMAGE_COUNT = 10000;

const FASHION_MNIST_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];

const CostKind = {
  CROSS_ENTROPY: "Cross_Entropy",
  MEAN_SQUARED_ERROR: "Mean_Squared_Error",
};

const ActivationKind = {
  SIGMOID: "Sigmoid",
  SOFTMAX: "Softmax",
  RELU: "ReLu",
  TANH: "Tanh",
};

const RandomKind = {
  GAUSSIAN: "Gaussian",
  STANDARD_NORMAL: "Standard_Normal",
  HE: "He",
};

const DEFAULT_SEED = Number(process.env.NN_SEED ?? 12345);

function createSeededRandom(seed) {
  let state = seed >>> 0;
  return function seededRandom() {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const random01 = createSeededRandom(DEFAULT_SEED);

function gaussianRandom(mean = 0, stdDev = 1) {
  let u = 0;
  let v = 0;
  while (u === 0) u = random01();
  while (v === 0) v = random01();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return mean + z * stdDev;
}

function argMax(values) {
  let idx = 0;
  let max = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
      idx = i;
    }
  }
  return idx;
}

function costFromKind(kind) {
  if (kind === CostKind.MEAN_SQUARED_ERROR) {
    return {
      kind,
      fn(ypred, y) {
        let cost = 0;
        for (let i = 0; i < y.length; i += 1) {
          const diff = ypred[i] - y[i];
          cost += diff * diff;
        }
        return 0.5 * cost;
      },
      derivative(pred, expected) {
        return pred - expected;
      },
    };
  }

  return {
    kind: CostKind.CROSS_ENTROPY,
    fn(pred, expected) {
      let cost = 0;
      for (let i = 0; i < pred.length; i += 1) {
        const a = pred[i];
        const y = expected[i];
        const c = -y * Math.log(a) - (1 - y) * Math.log(1 - a);
        cost += Number.isNaN(c) ? 0 : c;
      }

      return cost;
    },
    derivative(pred, expected) {
      if (pred === 0 || pred === 1) {
        return 0;
      }
      return (-pred + expected) / (pred * (pred - 1.0));
    },
  };
}

function activationFromKind(kind) {
  if (kind === ActivationKind.SIGMOID) {
    return {
      kind,
      fn(inputs, idx) {
        return 1.0 / (1.0 + Math.exp(-inputs[idx]));
      },
      derivative(inputs, idx) {
        const sig = 1.0 / (1.0 + Math.exp(-inputs[idx]));
        return sig * (1.0 - sig);
      },
    };
  }

  if (kind === ActivationKind.TANH) {
    return {
      kind,
      fn(inputs, idx) {
        return Math.tanh(inputs[idx]);
      },
      derivative(inputs, idx) {
        const cosh = Math.cosh(inputs[idx]);
        return 1.0 / (cosh * cosh);
      },
    };
  }

  if (kind === ActivationKind.SOFTMAX) {
    return {
      kind,
      fn(inputs, idx) {
        let sum = 0;
        for (let i = 0; i < inputs.length; i += 1) {
          sum += Math.exp(inputs[i]);
        }
        return Math.exp(inputs[idx]) / sum;
      },
      derivative(inputs, idx) {
        let sum = 0;
        for (let i = 0; i < inputs.length; i += 1) {
          sum += Math.exp(inputs[i]);
        }
        const x = Math.exp(inputs[idx]);
        return (x * sum - x * x) / (sum * sum);
      },
    };
  }

  return {
    kind: ActivationKind.RELU,
    fn(inputs, idx) {
      return Math.max(0, inputs[idx]);
    },
    derivative(inputs, idx) {
      return inputs[idx] > 0 ? 1.0 : 0.0;
    },
  };
}

function randomFromKind(kind) {
  if (kind === RandomKind.STANDARD_NORMAL) {
    return {
      kind,
      fn() {
        return gaussianRandom(0, 1);
      },
    };
  }

  if (kind === RandomKind.HE) {
    return {
      kind,
      fn(numIn) {
        return gaussianRandom(0, Math.sqrt(2.0 / numIn));
      },
    };
  }

  return {
    kind: RandomKind.GAUSSIAN,
    fn(numIn) {
      return gaussianRandom(0, 1.0 / Math.sqrt(numIn));
    },
  };
}

class Layer {
  constructor(numIn, numOut, activation, randomInit) {
    this.numIn = numIn;
    this.numOut = numOut;
    this.activation = activation;

    this.weights = Array.from({ length: numOut }, () => new Float32Array(numIn));
    this.weightGrads = Array.from({ length: numOut }, () => new Float32Array(numIn));
    this.biases = new Float32Array(numOut);
    this.biasGrads = new Float32Array(numOut);

    for (let i = 0; i < numOut; i += 1) {
      for (let j = 0; j < numIn; j += 1) {
        this.weights[i][j] = randomInit.fn(numIn, numOut);
      }
    }
  }

  calculateOutputLearn(input, learnData) {
    for (let i = 0; i < input.length; i += 1) {
      learnData.inputs[i] = input[i];
    }

    for (let neuron = 0; neuron < this.numOut; neuron += 1) {
      let weightedInput = this.biases[neuron];
      for (let i = 0; i < input.length; i += 1) {
        weightedInput += input[i] * this.weights[neuron][i];
      }
      learnData.weightedInputs[neuron] = weightedInput;
    }

    for (let i = 0; i < this.numOut; i += 1) {
      learnData.activations[i] = this.activation.fn(learnData.weightedInputs, i);
    }
  }

  calculateOutputNoLearn(input, output) {
    const weightedInputs = new Float32Array(this.numOut);

    for (let neuron = 0; neuron < this.numOut; neuron += 1) {
      let weightedInput = this.biases[neuron];
      for (let i = 0; i < input.length; i += 1) {
        weightedInput += input[i] * this.weights[neuron][i];
      }
      weightedInputs[neuron] = weightedInput;
    }

    for (let i = 0; i < this.numOut; i += 1) {
      output[i] = this.activation.fn(weightedInputs, i);
    }
  }
}

function createLearnData(layers) {
  return layers.map((layer) => ({
    inputs: new Float32Array(layer.numIn),
    activations: new Float32Array(layer.numOut),
    weightedInputs: new Float32Array(layer.numOut),
    nodeValues: new Float32Array(layer.numOut),
  }));
}

class NeuralNetwork {
  constructor() {
    this.layers = [];
    this.inputSize = 0;
    this.outputSize = 0;
    this.largestLayerSize = 0;
    this.random = randomFromKind(RandomKind.GAUSSIAN);
    this.cost = costFromKind(CostKind.CROSS_ENTROPY);
  }

  init(layerSizes, config) {
    this.largestLayerSize = Math.max(...layerSizes);
    this.inputSize = layerSizes[0];
    this.outputSize = layerSizes[layerSizes.length - 1];

    this.cost = costFromKind(config.cost);
    this.random = randomFromKind(config.random);

    this.layers = [];
    for (let i = 1; i < layerSizes.length; i += 1) {
      const numIn = layerSizes[i - 1];
      const numOut = layerSizes[i];
      const activationKind = i === layerSizes.length - 1 ? config.outputActivation : config.activation;
      this.layers.push(new Layer(numIn, numOut, activationFromKind(activationKind), this.random));
    }
  }

  forwardLearn(input, learnData) {
    let layerInput = input;
    for (let i = 0; i < this.layers.length; i += 1) {
      this.layers[i].calculateOutputLearn(layerInput, learnData[i]);
      layerInput = learnData[i].activations;
    }
  }

  forward(input) {
    let layerInput = Float32Array.from(input);
    let layerOutput = new Float32Array(this.largestLayerSize);

    for (let i = 0; i < this.layers.length; i += 1) {
      const layer = this.layers[i];
      const curIn = layerInput.subarray(0, layer.numIn);
      const curOut = layerOutput.subarray(0, layer.numOut);
      layer.calculateOutputNoLearn(curIn, curOut);
      layerInput = Float32Array.from(curOut);
      layerOutput = new Float32Array(this.largestLayerSize);
    }

    return Array.from(layerInput.subarray(0, this.outputSize));
  }

  updateGradients(expected, learnData) {
    const lastLayerIndex = this.layers.length - 1;
    const lastLayer = this.layers[lastLayerIndex];
    const lastLearn = learnData[lastLayerIndex];
    const cost = this.cost.fn(lastLearn.activations, expected);

    for (let i = 0; i < lastLayer.numOut; i += 1) {
      const costDerivative = this.cost.derivative(lastLearn.activations[i], expected[i]);
      const activationDerivative = lastLayer.activation.derivative(lastLearn.weightedInputs, i);
      lastLearn.nodeValues[i] = activationDerivative * costDerivative;
    }
    this.layerUpdateGradients(lastLayer, lastLearn);

    let oldNodeValues = lastLearn.nodeValues;
    for (let l = this.layers.length - 2; l >= 0; l -= 1) {
      const layer = this.layers[l];
      const nextLayer = this.layers[l + 1];
      const layerLearn = learnData[l];

      for (let i = 0; i < layer.numOut; i += 1) {
        let nodeValue = 0;
        for (let j = 0; j < oldNodeValues.length; j += 1) {
          const weightedInputDerivative = nextLayer.weights[j][i];
          nodeValue += weightedInputDerivative * oldNodeValues[j];
        }
        nodeValue *= layer.activation.derivative(layerLearn.weightedInputs, i);
        layerLearn.nodeValues[i] = nodeValue;
      }

      this.layerUpdateGradients(layer, layerLearn);
      oldNodeValues = layerLearn.nodeValues;
    }

    for (const data of learnData) {
      data.nodeValues.fill(0);
    }

    return cost;
  }

  layerUpdateGradients(layer, layerLearn) {
    for (let neuron = 0; neuron < layer.numOut; neuron += 1) {
      for (let j = 0; j < layer.numIn; j += 1) {
        const dcostDWeight = layerLearn.inputs[j] * layerLearn.nodeValues[neuron];
        layer.weightGrads[neuron][j] += dcostDWeight;
      }
      layer.biasGrads[neuron] += layerLearn.nodeValues[neuron];
    }
  }

  zeroGradients() {
    for (const layer of this.layers) {
      for (let i = 0; i < layer.numOut; i += 1) {
        layer.weightGrads[i].fill(0);
        layer.biasGrads[i] = 0;
      }
    }
  }

  countCorrect(batch) {
    let numCorrect = 0;
    for (const dataPoint of batch) {
      const output = this.forward(dataPoint.input);
      if (argMax(output) === dataPoint.label) {
        numCorrect += 1;
      }
    }

    return numCorrect;
  }

  serializeGradients() {
    return {
      layers: this.layers.map((layer) => ({
        weightGrads: layer.weightGrads.map((row) => Array.from(row)),
        biasGrads: Array.from(layer.biasGrads),
      })),
    };
  }

  absorbGradients(gradientPack) {
    for (let l = 0; l < this.layers.length; l += 1) {
      const layer = this.layers[l];
      const sourceLayer = gradientPack.layers[l];

      for (let i = 0; i < layer.numOut; i += 1) {
        for (let j = 0; j < layer.numIn; j += 1) {
          layer.weightGrads[i][j] += sourceLayer.weightGrads[i][j];
        }
        layer.biasGrads[i] += sourceLayer.biasGrads[i];
      }
    }
  }

  computeBatchGradients(batch) {
    this.zeroGradients();

    let totalCost = 0;
    const learnData = createLearnData(this.layers);

    for (const dataPoint of batch) {
      this.forwardLearn(dataPoint.input, learnData);
      totalCost += this.updateGradients(dataPoint.expected, learnData);
    }

    return {
      cost: totalCost,
      gradients: this.serializeGradients(),
    };
  }

  async learnParallel(batch, learnRate, regularization, workerPool) {
    if (!workerPool || workerPool.size <= 1 || batch.length < 2) {
      return this.learn(batch, learnRate, regularization);
    }

    const chunks = splitBatch(batch, workerPool.size);
    const modelSnapshot = this.toJSON();
    const tasks = chunks
      .filter((chunk) => chunk.length > 0)
      .map((chunk) => workerPool.runTask({ type: "learn", model: modelSnapshot, batch: chunk }));

    const results = await Promise.all(tasks);

    this.zeroGradients();
    let totalCost = 0;
    for (const result of results) {
      totalCost += result.cost;
      this.absorbGradients(result.gradients);
    }

    this.applyGradients(learnRate / batch.length, regularization);
    return totalCost / batch.length;
  }

  async evaluateParallel(batch, workerPool) {
    if (!workerPool || workerPool.size <= 1 || batch.length < 2) {
      return this.evaluate(batch);
    }

    const chunks = splitBatch(batch, workerPool.size);
    const modelSnapshot = this.toJSON();
    const tasks = chunks
      .filter((chunk) => chunk.length > 0)
      .map((chunk) => workerPool.runTask({ type: "evaluate", model: modelSnapshot, batch: chunk }));

    const results = await Promise.all(tasks);
    let numCorrect = 0;
    for (const result of results) {
      numCorrect += result.correct;
    }

    return numCorrect / batch.length;
  }

  applyGradients(learnRate, regularization) {
    const weightDecay = 1 - regularization * learnRate;

    for (const layer of this.layers) {
      for (let i = 0; i < layer.numOut; i += 1) {
        for (let j = 0; j < layer.numIn; j += 1) {
          const weight = weightDecay * layer.weights[i][j];
          layer.weights[i][j] = weight - learnRate * layer.weightGrads[i][j];
          layer.weightGrads[i][j] = 0;
        }

        layer.biases[i] += -learnRate * layer.biasGrads[i];
        layer.biasGrads[i] = 0;
      }
    }
  }

  learn(batch, learnRate, regularization = 0) {
    let totalCost = 0;
    const learnData = createLearnData(this.layers);

    for (const dataPoint of batch) {
      this.forwardLearn(dataPoint.input, learnData);
      totalCost += this.updateGradients(dataPoint.expected, learnData);
    }

    this.applyGradients(learnRate / batch.length, regularization);
    return totalCost / batch.length;
  }

  evaluate(batch) {
    let numCorrect = 0;
    for (const dataPoint of batch) {
      const output = this.forward(dataPoint.input);
      if (argMax(output) === dataPoint.label) {
        numCorrect += 1;
      }
    }

    return numCorrect / batch.length;
  }

  predict(dataPoint) {
    const output = this.forward(dataPoint.input);
    const prediction = argMax(output);
    return {
      prediction,
      confidence: output[prediction],
    };
  }

  toJSON() {
    return {
      inputSize: this.inputSize,
      outputSize: this.outputSize,
      largestLayerSize: this.largestLayerSize,
      costKind: this.cost.kind,
      randomKind: this.random.kind,
      layers: this.layers.map((layer) => ({
        numIn: layer.numIn,
        numOut: layer.numOut,
        activationKind: layer.activation.kind,
        weights: layer.weights.map((w) => Array.from(w)),
        biases: Array.from(layer.biases),
      })),
    };
  }

  static fromJSON(data) {
    const network = new NeuralNetwork();
    network.inputSize = data.inputSize;
    network.outputSize = data.outputSize;
    network.largestLayerSize = data.largestLayerSize;
    network.cost = costFromKind(data.costKind);
    network.random = randomFromKind(data.randomKind);

    network.layers = data.layers.map((layerData) => {
      const layer = new Layer(layerData.numIn, layerData.numOut, activationFromKind(layerData.activationKind), network.random);
      layer.weights = layerData.weights.map((row) => Float32Array.from(row));
      layer.weightGrads = Array.from({ length: layer.numOut }, () => new Float32Array(layer.numIn));
      layer.biases = Float32Array.from(layerData.biases);
      layer.biasGrads = new Float32Array(layer.numOut);
      return layer;
    });

    return network;
  }
}

function batchCreate(inputs, labels, numLabels) {
  const batch = new Array(inputs.length);
  for (let i = 0; i < inputs.length; i += 1) {
    const expected = new Float32Array(numLabels);
    expected[labels[i]] = 1.0;
    batch[i] = {
      input: Float32Array.from(inputs[i]),
      expected,
      label: labels[i],
    };
  }
  return batch;
}

function loadMnistImages(filePath) {
  const data = fs.readFileSync(filePath);
  const magic = data.readInt32BE(0);
  if (magic !== 2051) {
    throw new Error(`Magic number must be 2051: ${filePath}`);
  }

  const count = data.readInt32BE(4);
  const rows = data.readInt32BE(8);
  const cols = data.readInt32BE(12);
  const imageSize = rows * cols;

  const images = new Array(count);
  let offset = 16;

  for (let i = 0; i < count; i += 1) {
    const image = new Float32Array(imageSize);
    for (let j = 0; j < imageSize; j += 1) {
      image[j] = data[offset] / 255.0;
      offset += 1;
    }
    images[i] = image;
  }

  return images;
}

function loadMnistLabels(filePath) {
  const data = fs.readFileSync(filePath);
  const magic = data.readInt32BE(0);
  if (magic !== 2049) {
    throw new Error(`Magic number must be 2049: ${filePath}`);
  }

  const count = data.readInt32BE(4);
  const labels = new Array(count);
  for (let i = 0; i < count; i += 1) {
    labels[i] = data[8 + i];
  }

  return labels;
}

function loadMnist(dir) {
  const trainLabelsPath = path.join(dir, "train-labels-idx1-ubyte");
  const trainImagesPath = path.join(dir, "train-images-idx3-ubyte");
  const testLabelsPath = path.join(dir, "t10k-labels-idx1-ubyte");
  const testImagesPath = path.join(dir, "t10k-images-idx3-ubyte");

  const trainLabels = loadMnistLabels(trainLabelsPath);
  const trainImages = loadMnistImages(trainImagesPath);
  const trainBatch = batchCreate(trainImages, trainLabels, 10);

  const testLabels = loadMnistLabels(testLabelsPath);
  const testImages = loadMnistImages(testImagesPath);
  const testBatch = batchCreate(testImages, testLabels, 10);

  const trainSet = {
    data: trainBatch,
    classes: null,
    inputSize: trainBatch[0].input.length,
    outputSize: trainBatch[0].expected.length,
  };
  const testSet = {
    data: testBatch,
    classes: null,
    inputSize: testBatch[0].input.length,
    outputSize: testBatch[0].expected.length,
  };

  return { trainSet, testSet };
}

function loadCifarBatch(filePath, numLabels) {
  const data = fs.readFileSync(filePath);
  const points = [];
  const imageBytes = 3 * CIFAR_RES * CIFAR_RES;
  const recordSize = imageBytes + 1;

  let offset = 0;
  while (offset + recordSize <= data.length) {
    const label = data[offset];
    offset += 1;

    const input = new Float32Array(imageBytes);
    for (let i = 0; i < imageBytes; i += 1) {
      input[i] = data[offset + i] / 255.0;
    }
    offset += imageBytes;

    const expected = new Float32Array(numLabels);
    expected[label] = 1.0;
    points.push({ input, expected, label });
  }

  return points;
}

function loadCifar(dir) {
  const metaPath = path.join(dir, "batches.meta.txt");
  const classes = fs.readFileSync(metaPath, "utf-8").trim().split(/\r?\n/).filter(Boolean);

  const numLabels = classes.length;
  const trainBatch = [];

  for (let i = 1; i <= 5; i += 1) {
    const batchPath = path.join(dir, `data_batch_${i}.bin`);
    if (fs.existsSync(batchPath)) {
      trainBatch.push(...loadCifarBatch(batchPath, numLabels));
    }
  }

  const testPath = path.join(dir, "test_batch.bin");
  const testBatch = fs.existsSync(testPath) ? loadCifarBatch(testPath, numLabels) : [];

  if (trainBatch.length === 0 || testBatch.length === 0) {
    throw new Error("CIFAR-10 binary files were not found (data_batch_*.bin / test_batch.bin)");
  }

  const trainSet = {
    data: trainBatch,
    classes,
    inputSize: trainBatch[0].input.length,
    outputSize: trainBatch[0].expected.length,
  };
  const testSet = {
    data: testBatch,
    classes,
    inputSize: testBatch[0].input.length,
    outputSize: testBatch[0].expected.length,
  };

  return { trainSet, testSet };
}

function saveModel(model, filePath) {
  fs.writeFileSync(filePath, JSON.stringify(model.toJSON()));
}

function loadModel(filePath) {
  const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  return NeuralNetwork.fromJSON(data);
}

function shuffleInPlace(array) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random01() * (i + 1));
    const tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
}

function splitBatch(batch, parts) {
  const chunks = [];
  const chunkSize = Math.ceil(batch.length / parts);
  for (let i = 0; i < batch.length; i += chunkSize) {
    chunks.push(batch.slice(i, i + chunkSize));
  }
  return chunks;
}

class WorkerPool {
  constructor(size) {
    this.size = Math.max(1, size);
    this.workers = [];
    this.pending = new Map();
    this.nextTaskId = 1;

    for (let i = 0; i < this.size; i += 1) {
      const worker = new Worker(__filename);
      worker.on("message", (message) => {
        const pendingTask = this.pending.get(message.id);
        if (!pendingTask) {
          return;
        }

        this.pending.delete(message.id);
        if (message.error) {
          pendingTask.reject(new Error(message.error));
          return;
        }

        pendingTask.resolve(message.result);
      });

      worker.on("error", (error) => {
        for (const pendingTask of this.pending.values()) {
          pendingTask.reject(error);
        }
        this.pending.clear();
      });

      this.workers.push(worker);
    }
  }

  runTask(payload) {
    const id = this.nextTaskId;
    this.nextTaskId += 1;
    const worker = this.workers[(id - 1) % this.workers.length];

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ id, ...payload });
    });
  }

  async close() {
    await Promise.all(this.workers.map((worker) => worker.terminate()));
  }
}

function startWorker() {
  parentPort.on("message", (message) => {
    try {
      const network = NeuralNetwork.fromJSON(message.model);

      if (message.type === "learn") {
        const result = network.computeBatchGradients(message.batch);
        parentPort.postMessage({ id: message.id, result });
        return;
      }

      if (message.type === "evaluate") {
        const correct = network.countCorrect(message.batch);
        parentPort.postMessage({ id: message.id, result: { correct } });
        return;
      }

      throw new Error(`Unknown worker task type: ${message.type}`);
    } catch (error) {
      parentPort.postMessage({ id: message.id, error: error.message });
    }
  });
}

function getDatasetPreset(kind, inputSize, outputSize) {
  if (kind === "Fashion") {
    return {
      layerSizes: [inputSize, 10, outputSize],
      config: {
        cost: CostKind.CROSS_ENTROPY,
        activation: ActivationKind.RELU,
        outputActivation: ActivationKind.SOFTMAX,
        random: RandomKind.HE,
      },
      trainSplit: 0.9,
      miniBatchSize: 256,
      learnRate: 0.005,
      regularization: 0.0005,
      epochs: 8,
    };
  }

  if (kind === "Digits") {
    return {
      layerSizes: [inputSize, 16, outputSize],
      config: {
        cost: CostKind.CROSS_ENTROPY,
        activation: ActivationKind.RELU,
        outputActivation: ActivationKind.SOFTMAX,
        random: RandomKind.HE,
      },
      trainSplit: 0.85,
      miniBatchSize: 256,
      learnRate: 0.01,
      regularization: 0.0005,
      epochs: 5,
    };
  }

  return {
    layerSizes: [inputSize, 1024, 512, outputSize],
    config: {
      cost: CostKind.CROSS_ENTROPY,
      activation: ActivationKind.RELU,
      outputActivation: ActivationKind.SOFTMAX,
      random: RandomKind.HE,
    },
    trainSplit: 0.85,
    miniBatchSize: 128,
    learnRate: 0.003,
    regularization: 0.01,
    epochs: 40,
  };
}

function modelMatchesPreset(model, preset) {
  if (!model || model.inputSize !== preset.layerSizes[0] || model.outputSize !== preset.layerSizes[preset.layerSizes.length - 1]) {
    return false;
  }

  if (model.layers.length !== preset.layerSizes.length - 1) {
    return false;
  }

  for (let i = 0; i < model.layers.length; i += 1) {
    const layer = model.layers[i];
    if (layer.numIn !== preset.layerSizes[i] || layer.numOut !== preset.layerSizes[i + 1]) {
      return false;
    }
  }

  return true;
}

function runViewerConsole(model, dataSet, kind, sampleCount = 20) {
  console.log(`\nViewer (${kind}) - showing ${Math.min(sampleCount, dataSet.data.length)} samples:`);
  for (let i = 0; i < Math.min(sampleCount, dataSet.data.length); i += 1) {
    const dp = dataSet.data[i];
    const { prediction, confidence } = model.predict(dp);

    const predText = dataSet.classes ? dataSet.classes[prediction] : String(prediction);
    const labelText = dataSet.classes ? dataSet.classes[dp.label] : String(dp.label);
    const verdict = prediction === dp.label ? "Correct" : "Incorrect";
    console.log(`[${i}] ${verdict} | Prediction: ${predText} (${(100 * confidence).toFixed(2)}%) | Label: ${labelText}`);
  }
}

async function main() {
  let dataSetKind = "Digits";
  let dataSetDir = "digits-mnist";

  if (process.argv.length > 2) {
    const requested = process.argv[2];
    switch (requested) {
      case "digits":
        dataSetKind = "Digits";
        dataSetDir = "digits-mnist";
        break;
      case "fashion":
        dataSetKind = "Fashion";
        dataSetDir = "fashion-mnist";
        break;
      case "cifar":
        dataSetKind = "Cifar";
        dataSetDir = "cifar-10";
        break;
      default:
        console.error(`usage: node main.js <dataset>`);
        console.error("Supported datasets: digits, fashion, cifar");
        process.exit(1);
    }
  }

  console.log("Loading Dataset");
  let trainSet;
  let testSet;
  const dataPath = path.join(__dirname, dataSetDir);
  console.log(os.cpus().length, "CPU cores detected");
  const workerCount = Math.max(
    1,
    Math.min(typeof os.availableParallelism === "function" ? os.availableParallelism() : os.cpus().length, os.cpus().length),
  );
  const workerPool = new WorkerPool(workerCount);

  try {
    if (dataSetKind === "Digits" || dataSetKind === "Fashion") {
      ({ trainSet, testSet } = loadMnist(dataPath));
      if (dataSetKind === "Fashion") {
        testSet.classes = FASHION_MNIST_CLASSES;
      }
    } else {
      ({ trainSet, testSet } = loadCifar(dataPath));
    }

    const preset = getDatasetPreset(dataSetKind, trainSet.inputSize, trainSet.outputSize);
    console.log("Training Network (fresh run)");
    const model = new NeuralNetwork();
    model.init(preset.layerSizes, preset.config);

    const splitIdx = Math.floor(preset.trainSplit * trainSet.data.length);
    const train = trainSet.data.slice(0, splitIdx);
    const validation = trainSet.data.slice(splitIdx);

    const miniBatchSize = preset.miniBatchSize;
    const learnRate = preset.learnRate;
    const regularization = preset.regularization;
    const epochs = preset.epochs;

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      shuffleInPlace(train);
      let cost = 0;
      for (let i = 0; i + miniBatchSize <= train.length; i += miniBatchSize) {
        const batch = train.slice(i, i + miniBatchSize);
        cost = await model.learnParallel(batch, learnRate, regularization, workerPool);
      }

      const evalAccuracy = await model.evaluateParallel(validation, workerPool);
      console.log(`Epoch(${epoch + 1}) Accuracy = ${(100 * evalAccuracy).toFixed(2)}% | Batch Cost = ${cost.toFixed(6)}`);
    }

    console.log("Testing");
    const testAccuracy = await model.evaluateParallel(testSet.data, workerPool);
    console.log(`Accuracy on test set: ${(100 * testAccuracy).toFixed(2)}%`);

    runViewerConsole(model, testSet, dataSetKind);
  } finally {
    await workerPool.close();
  }
}

if (!isMainThread) {
  startWorker();
} else if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
