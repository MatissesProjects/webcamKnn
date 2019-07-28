const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
let net;
var buttonDown;
var countOfClassImages = [0, 0, 0, 0];

const imagesPerSecond = 10;
const DELAY_TIME = 1000 / imagesPerSecond;

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia ||
                             navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject()
      );
    } else {
      reject();
    }
  });
}

function updateButtonNames() {
  document.getElementById('classA').innerText = `Add ${document.getElementById('textA').value}`;
  document.getElementById('classB').innerText = `Add ${document.getElementById('textB').value}`;
  document.getElementById('classC').innerText = `Add ${document.getElementById('textC').value}`;
  document.getElementById('classD').innerText = `Add ${document.getElementById('textD').value}`;
}

function addClickListeners(addExample) { // While clicking a button, add an example every .1 second for that class.
  const createListener = (className, classLabelNumber) => {
    const element = document.getElementById(className);
    element.addEventListener('mousedown', () => {buttonDown = setInterval(() => {addExample(classLabelNumber);}, DELAY_TIME);});
    element.addEventListener('mouseup', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
    element.addEventListener('mouseout', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
  }
  createListener('classA', 0);
  createListener('classB', 1);
  createListener('classC', 2);
  createListener('classD', 3);
}

function updateCountOfClassExamples() {
  document.getElementById('labelA').innerText = countOfClassImages[0];
  document.getElementById('labelB').innerText = countOfClassImages[1];
  document.getElementById('labelC').innerText = countOfClassImages[2];
  document.getElementById('labelD').innerText = countOfClassImages[3];
}

async function app() {
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();
  // Reads an image from the webcam and associates it with a specific class index.
  const addExample = (classId) => {
    countOfClassImages[classId]++; // for each image captured keep count so we can display
    const activation = net.infer(webcamElement, 'conv_preds');
    classifier.addExample(activation, classId); // Pass the intermediate activation to the classifier.
    updateCountOfClassExamples();
    console.log(`added ${classId} ${activation}`);
  };

  addClickListeners(addExample);
  document.getElementById('saveButton').addEventListener('click', () => {save();});
  document.getElementById('loadButton').addEventListener('click', () => {load();});

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const activation = net.infer(webcamElement, 'conv_preds'); // Get the activation from mobilenet from the webcam.
      const result = await classifier.predictClass(activation);
      const classNames = [document.getElementById('textA').value, document.getElementById('textB').value,
                          document.getElementById('textC').value, document.getElementById('textD').value];
      document.getElementById('console').innerText = `
        prediction: ${classNames[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
      updateButtonNames();
    }
    await tf.nextFrame();
  }
}

function save() {
   let dataset = classifier.getClassifierDataset()
   var datasetObj = {}
   Object.keys(dataset).forEach((key) => {
     let data = dataset[key].dataSync();
     // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...]
     // instead of object e.g {0:"0.1", 1:"-0.2"...}
     datasetObj[key] = Array.from(data);
   });
   let jsonStr = JSON.stringify(datasetObj)
   //can be change to other source
   // localStorage.setItem("myData", jsonStr);
   // location.href = "data:application/octet-stream," + encodeURIComponent(jsonStr);
   document.getElementById('link').href = "data:application/octet-stream," + encodeURIComponent(jsonStr);
   document.getElementById('link').click();
   // classifier.save('downloads://test')
 }

 function load() {
    let loadButton = document.getElementById('loadButton');
    if ('files' in loadButton) {
      if (loadButton.files.length == 0) {
        txt = "Select one or more files.";
      } else {
        var readerData;
        var reader = new FileReader();
        reader.onload = fileEvent => readerData = fileEvent.target.result;
        reader.readAsText(loadButton.files[0]);
        let dataset = readerData;
        let tensorObj = JSON.parse(dataset)
        //covert back to tensor
        Object.keys(tensorObj).forEach((key) => {
          tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1000, 1000])
        })
        classifier.setClassifierDataset(tensorObj);
      }
    }
     //can be change to other source
    // let dataset = localStorage.getItem("myData")
    // let tensorObj = JSON.parse(dataset)
    // //covert back to tensor
    // Object.keys(tensorObj).forEach((key) => {
    //   tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1000, 1000])
    // })
    // classifier.setClassifierDataset(tensorObj);
  }

app();
