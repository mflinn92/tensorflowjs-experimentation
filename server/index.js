const tf = require('@tensorflow/tfjs-node');
const mobileNet = require('@tensorflow-models/mobilenet');
const jpeg = require('jpeg-js');
const fs = require('fs');
const util = require('util');
const path = require('path');

const readFile = util.promisify(fs.readFile);

const getImgTensor = async (filePath) => {
  let jpegData = await readFile(filePath);
  let tensor = tf.node.decodeImage(jpegData);
  return tensor
}


const Classify = async() => {
  const model = await mobileNet.load();
  const tensor = await getImgTensor('eagle.JPEG');
  return await model.classify(tensor);

}

Classify()
  .then(prediction => console.log(prediction))


// const decodeImage = async(rawJpegData) => {
//   return tf.node.decodeImage(rawJpegData);
// }
// getImgBuffer('eagle.JPEG')
//   .then((imgData) => decodeImage(imgData.data))
//   .then((tensor) => console.log(tensor));

// const model = mobileNet.load()
//   .then(console.log('model loaded'));