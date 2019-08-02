const webcamElement = document.getElementById('webcam');
const imageClassifier1 = knnClassifier.create();
const imageClassifier2 = knnClassifier.create();
const classifier3 = knnClassifier.create();
let net;
let buttonDown;
let countOfClassImages = [0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0, 0, 0];

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

async function updateClassifierConsole(classifier, consoleNumber, activation) {
  if (classifier.getNumClasses() > 0) {
    const result = await classifier.predictClass(activation);
    const classNames = [document.getElementById(`text${consoleNumber}-1`).value, document.getElementById(`text${consoleNumber}-2`).value,
                        document.getElementById(`text${consoleNumber}-3`).value, document.getElementById(`text${consoleNumber}-4`).value];
    document.getElementById(`console${consoleNumber}`).innerText = `
      prediction: ${classNames[parseInt(result.label)]}\n
      probability: ${result.confidences[parseInt(result.label)]}
    `;
    return [parseInt(result.label), result.confidences[parseInt(result.label)]]

  }
  return []
}

function updateCountOfClassExamples() {
  document.getElementById('label1-1').innerText = countOfClassImages[0];
  document.getElementById('label1-2').innerText = countOfClassImages[1];
  document.getElementById('label1-3').innerText = countOfClassImages[2];
  document.getElementById('label1-4').innerText = countOfClassImages[3];
  document.getElementById('label2-1').innerText = countOfClassImages[4];
  document.getElementById('label2-2').innerText = countOfClassImages[5];
  document.getElementById('label2-3').innerText = countOfClassImages[6];
  document.getElementById('label2-4').innerText = countOfClassImages[7];
  document.getElementById('label3-1').innerText = countOfClassImages[8];
  document.getElementById('label3-2').innerText = countOfClassImages[9];
  document.getElementById('label3-3').innerText = countOfClassImages[10];
  document.getElementById('label3-4').innerText = countOfClassImages[11];
}

function updateButtonNames() {
  document.getElementById('class1-1').innerText = `Add ${document.getElementById('text1-1').value}`;
  document.getElementById('class1-2').innerText = `Add ${document.getElementById('text1-2').value}`;
  document.getElementById('class1-3').innerText = `Add ${document.getElementById('text1-3').value}`;
  document.getElementById('class1-4').innerText = `Add ${document.getElementById('text1-4').value}`;
  document.getElementById('class2-1').innerText = `Add ${document.getElementById('text2-1').value}`;
  document.getElementById('class2-2').innerText = `Add ${document.getElementById('text2-2').value}`;
  document.getElementById('class2-3').innerText = `Add ${document.getElementById('text2-3').value}`;
  document.getElementById('class2-4').innerText = `Add ${document.getElementById('text2-4').value}`;
  document.getElementById('class3-1').innerText = `Add ${document.getElementById('text3-1').value}`;
  document.getElementById('class3-2').innerText = `Add ${document.getElementById('text3-2').value}`;
  document.getElementById('class3-3').innerText = `Add ${document.getElementById('text3-3').value}`;
  document.getElementById('class3-4').innerText = `Add ${document.getElementById('text3-4').value}`;
}

function addClickListeners(addExample) { // While clicking a button, add an example every .1 second for that class.
  const createListener = (className, classLabelNumber) => {
    const element = document.getElementById(className);
    element.addEventListener('mousedown', () => {buttonDown = setInterval(() => {addExample(classLabelNumber);}, DELAY_TIME);});
    element.addEventListener('mouseup', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
    element.addEventListener('mouseout', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
  }
  createListener('class1-1', 0);
  createListener('class1-2', 1);
  createListener('class1-3', 2);
  createListener('class1-4', 3);
  createListener('class2-1', 4);
  createListener('class2-2', 5);
  createListener('class2-3', 6);
  createListener('class2-4', 7);
  createListener('class3-1', 8);
  createListener('class3-2', 9);
  createListener('class3-3', 10);
  createListener('class3-4', 11);
}

async function app() {
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');
  let listenersAdded = false;

  await setupWebcam();
  // Reads an image from the webcam and associates it with a specific class index.
  const addExample = async (classId) => {
    countOfClassImages[classId]++; // for each image captured keep count so we can display
    const activation = net.infer(webcamElement, 'conv_preds');
    if(classId<4){ console.log('added to classifier1');imageClassifier1.addExample(activation, classId); } // Pass the intermediate activation to the classifier.
    if(classId>=4 && classId<8) { console.log('added to classifier2');imageClassifier2.addExample(activation, classId-4); } // Pass the intermediate activation to the classifier.
    if(classId>=8) {
      results1 = await updateClassifierConsole(imageClassifier1, '1', activation);
      results2 = await updateClassifierConsole(imageClassifier2, '2', activation);
      classifier3.addExample(tf.tensor([results1, results2]), classId-8);
    }
    updateCountOfClassExamples();
    // console.log(`added ${classId} ${activation}`);
  };

  while (true) {
    if(!listenersAdded){
      addClickListeners(addExample);
      document.getElementById('saveButton').addEventListener('click', () => {save();});
      document.getElementById('loadButton').addEventListener('click', () => {load();});
      listenersAdded = true;
    }
    const webcamActivation = net.infer(webcamElement, 'conv_preds'); // Get the activation from mobilenet from the webcam.
    results1 = await updateClassifierConsole(imageClassifier1, '1', webcamActivation);
    results2 = await updateClassifierConsole(imageClassifier2, '2', webcamActivation);
    updateClassifierConsole(classifier3, '3', tf.tensor([results1, results2]));
    updateButtonNames();
    await tf.nextFrame();
  }
}

function save() {
  let dataset1 = imageClassifier1.getClassifierDataset()
  let dataset2 = imageClassifier2.getClassifierDataset() // TODO: ADD
  let dataset3 = classifier3.getClassifierDataset() // TODO: ADD
  let datasetObj = Object.keys(dataset1).map((key) => {
     let data = dataset[key].dataSync();
     // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...]
     // instead of object e.g {0:"0.1", 1:"-0.2"...}
     return Array.from(data);
   });
   // Object.keys(dataset).forEach((key) => {
   //   let data = dataset[key].dataSync();
   //   // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...]
   //   // instead of object e.g {0:"0.1", 1:"-0.2"...}
   //   datasetObj[key] = Array.from(data);
   // });
   let jsonStr = JSON.stringify(datasetObj)
   //can be change to other source
   // localStorage.setItem("myData", jsonStr);
   // location.href = "data:application/octet-stream," + encodeURIComponent(jsonStr);
   document.getElementById('link').href = "data:application/octet-stream," + encodeURIComponent(jsonStr);
   document.getElementById('link').click();
   // classifier.save('downloads://test')
 }

 function load() {
    console.log('calling load');
    let loadButton = document.getElementById('loadButton');
    if ('files' in loadButton) {
      console.log('files in load');
      if (loadButton.files.length == 0) {
        txt = "Select one or more files.";
      } else {
        console.log('has length');
        let readerData;
        let reader = new FileReader();
        reader.onload = fileEvent => readerData = fileEvent.target.result;
        reader.readAsText(loadButton.files[0]);
        let dataset = readerData;
        console.log(dataset);
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
