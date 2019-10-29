const tf = require('@tensorflow/tfjs-node');
const mobileNet = require('@tensorflow-models/mobilenet');
const fs = require('fs');
const util = require('util');


const readFile = util.promisify(fs.readFile);

const getImgTensor = async (filePath) => {
  let jpegData = await readFile(filePath);
  return tf.node.decodeImage(jpegData);
}


const Classify = async () => {
  const model = await mobileNet.load();
  const tensor = await getImgTensor('eagle.JPEG');
  return await model.classify(tensor);
}

Classify()
  .then(prediction => console.log(prediction))
