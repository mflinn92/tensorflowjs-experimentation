let model;

let App = async() => {
  console.log('loading mobileNet');

  model = await mobilenet.load();
  console.log('Model loaded');

  const picture = document.getElementById('img');
  const prediction = await model.classify(picture);
  console.log(prediction);
}

App();