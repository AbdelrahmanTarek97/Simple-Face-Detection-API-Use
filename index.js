const video = document.getElementById("video");

// Request user's permission to use webcam, and assign webcam stream as the source of the video
const startVideo = () => {
  var stream = navigator.getUserMedia(
    { video: true },
    stream => (video.srcObject = stream),
    err => console.error(err)
  );
};

// Load model weights for the used neural network
const loadModels = async () => {
  await faceapi.nets.tinyFaceDetector.loadFromUri("./face-api/weights/");
  await faceapi.nets.faceLandmark68Net.loadFromUri("./face-api/weights/");
  await faceapi.nets.faceRecognitionNet.loadFromUri("./face-api/weights/");
  await faceapi.nets.faceExpressionNet.loadFromUri("./face-api/weights/");
  await faceapi.nets.ssdMobilenetv1.loadFromUri("./face-api/weights/");
};

// This function is called whenever a new face detection is needed
const detectFaces = () => {
  return faceapi
    .detectAllFaces(video)
    .withFaceLandmarks()
    .withFaceExpressions();
};

const mainFunction = async () => {
  await loadModels();
  startVideo();
};

video.addEventListener("play", () => {
  // create a canvas from the video element, which is like taking a screenshot
  const canvas = faceapi.createCanvasFromMedia(video);
  // append the canvas to the html
  document.body.append(canvas);
  // displaySize of the video element
  const displaySize = { width: 720, height: 560 };
  // make the canvas the same size of the video element
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    // clear the canvas from last detection
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    // detect the face with features and expressions
    const detectionWithExpressions = await detectFaces();
    // if there is a detection, draw it on the canvas
    if (detectionWithExpressions) {
      const resizedDetections = faceapi.resizeResults(
        detectionWithExpressions,
        displaySize
      );
      faceapi.draw.drawDetections(canvas, resizedDetections);
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
    }
  }, 100);
});

// run the main function
mainFunction();
