const webcamElement = document.getElementById('webcam');
const imageClassifier1 = knnClassifier.create();
const imageClassifier2 = knnClassifier.create();
const classifier3 = knnClassifier.create();
let net;
let buttonDown;
let countOfClassImages = [];

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
  let numberOfClassifierClasses = document.getElementById(`numberOfClassifier${consoleNumber}Classes`).innerText;
  let classNames = [];
  if (classifier.getNumClasses() > 0) {
    const result = await classifier.predictClass(activation);
    for( let i=0; i < numberOfClassifierClasses; ++i) { classNames.push(document.getElementById(`text${consoleNumber}-${i+1}`).value); }
    document.getElementById(`console${consoleNumber}`).innerText = `
      prediction: ${classNames[parseInt(result.label)]}\n
      probability: ${result.confidences[parseInt(result.label)]}
    `;
    return [parseInt(result.label), result.confidences[parseInt(result.label)]]

  }
  return []
}

function updateCountOfClassExamples() {
  let numberOfClassifier1Classes = parseInt(document.getElementById('numberOfClassifier1Classes').innerText);
  let numberOfClassifier2Classes = parseInt(document.getElementById('numberOfClassifier2Classes').innerText);
  let numberOfClassifier3Classes = parseInt(document.getElementById('numberOfClassifier3Classes').innerText);
  for (let i = 0; i < numberOfClassifier1Classes; i++) {
    document.getElementById(`label1-${i+1}`).innerText = countOfClassImages[i];
  }
  for (let i = 0; i < numberOfClassifier2Classes; i++) {
    document.getElementById(`label2-${i+1}`).innerText = countOfClassImages[i+numberOfClassifier1Classes];
  }
  for (let i = 0; i < numberOfClassifier3Classes; i++) {
    document.getElementById(`label3-${i+1}`).innerText = countOfClassImages[i+numberOfClassifier1Classes+numberOfClassifier2Classes];
  }
}

function updateButtonNames() {
    let numberOfClassifier1Classes = parseInt(document.getElementById('numberOfClassifier1Classes').innerText);
    let numberOfClassifier2Classes = parseInt(document.getElementById('numberOfClassifier2Classes').innerText);
    let numberOfClassifier3Classes = parseInt(document.getElementById('numberOfClassifier3Classes').innerText);
    for (let i = 0; i < numberOfClassifier1Classes; i++) {
      let textToGrab = document.getElementById(`text1-${i+1}`).value;
      document.getElementById(`class1-${i+1}`).innerText = `Add ${textToGrab}`;
    }
    for (let i = 0; i < numberOfClassifier2Classes; i++) {
      let textToGrab = document.getElementById(`text2-${i+1}`).value;
      document.getElementById(`class2-${i+1}`).innerText = `Add ${textToGrab}`;
    }
    for (let i = 0; i < numberOfClassifier3Classes; i++) {
      let textToGrab = document.getElementById(`text3-${i+1}`).value;
      document.getElementById(`class3-${i+1}`).innerText = `Add ${textToGrab}`;
    }
}

function addClickListeners(addExample) { // While clicking a button, add an example every .1 second for that class.
  const createListener = (className, classLabelNumber) => {
    const element = document.getElementById(className);
    element.addEventListener('mousedown', () => {buttonDown = setInterval(() => {addExample(classLabelNumber);}, DELAY_TIME);});
    element.addEventListener('mouseup', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
    element.addEventListener('mouseout', () => {console.log(`released ${className}`); clearInterval(buttonDown)});
  }

  let numberOfClassifier1Classes = parseInt(document.getElementById('numberOfClassifier1Classes').innerText);
  let numberOfClassifier2Classes = parseInt(document.getElementById('numberOfClassifier2Classes').innerText);
  let numberOfClassifier3Classes = parseInt(document.getElementById('numberOfClassifier3Classes').innerText);
  for (let i = 0; i < numberOfClassifier1Classes; i++) {
    let textToGrab = document.getElementById(`text1-${i+1}`).value;
    createListener(`class1-${i+1}`, i);
  }
  for (let i = 0; i < numberOfClassifier2Classes; i++) {
    let textToGrab = document.getElementById(`text2-${i+1}`).value;
    createListener(`class2-${i+1}`, i+numberOfClassifier1Classes);
  }
  for (let i = 0; i < numberOfClassifier3Classes; i++) {
    let textToGrab = document.getElementById(`text3-${i+1}`).value;
    createListener(`class3-${i+1}`, i+numberOfClassifier1Classes+numberOfClassifier2Classes);
  }
}

async function app() {
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');
  let listenersAdded = false;

  await setupWebcam();
  // Reads an image from the webcam and associates it with a specific class index.
  const addExample = async (classId) => {
    let numberOfClassifier1Classes = parseInt(document.getElementById('numberOfClassifier1Classes').innerText);
    let numberOfClassifier2Classes = parseInt(document.getElementById('numberOfClassifier2Classes').innerText);
    countOfClassImages[classId]++; // for each image captured keep count so we can display
    const activation = net.infer(webcamElement, 'conv_preds');
    if(classId<numberOfClassifier1Classes){ console.log('added to classifier1');imageClassifier1.addExample(activation, classId); } // Pass the intermediate activation to the classifier.
    if(classId>=numberOfClassifier1Classes && classId<numberOfClassifier2Classes+numberOfClassifier1Classes) { console.log('added to classifier2');imageClassifier2.addExample(activation, classId-numberOfClassifier1Classes); } // Pass the intermediate activation to the classifier.
    if(classId>=numberOfClassifier2Classes+numberOfClassifier1Classes) {
      results1 = await updateClassifierConsole(imageClassifier1, '1', activation);
      results2 = await updateClassifierConsole(imageClassifier2, '2', activation);
      classifier3.addExample(tf.tensor([results1, results2]), classId-numberOfClassifier2Classes-numberOfClassifier1Classes);
    }
    updateCountOfClassExamples();
    // console.log(`added ${classId} ${activation}`);
  };

  while (true) {
    if(!listenersAdded){
      addClickListeners(addExample);
      initilizeClassCount();
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

function initilizeClassCount() {
  let numberOfClassifier1Classes = parseInt(document.getElementById('numberOfClassifier1Classes').innerText);
  let numberOfClassifier2Classes = parseInt(document.getElementById('numberOfClassifier2Classes').innerText);
  let numberOfClassifier3Classes = parseInt(document.getElementById('numberOfClassifier3Classes').innerText);

  countOfClassImages= Array(numberOfClassifier1Classes + numberOfClassifier2Classes + numberOfClassifier3Classes).fill(0);
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
